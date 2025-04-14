"""
exchange_risk_limits.py - Exchange-specific risk limits

This module provides exchange-specific risk limits and validation functionality,
ensuring that orders meet the requirements of the target exchange and do not
exceed exchange-imposed limits. It implements adaptive rate limiting and
dynamic exchange constraints based on market conditions.

Each exchange has its own specific limits on order sizes, notional values,
rate limitations, and other constraints that must be enforced before
submitting orders.
"""

import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass
import time
import json
import os
from pathlib import Path

# Configure logger
logger = logging.getLogger(__name__)


class RateLimitType(Enum):
    """Rate limit types"""
    ORDERS = "orders"
    REQUESTS = "requests"
    VOLUME = "volume"
    NOTIONAL = "notional"
    CANCEL = "cancel"


class ExchangeRiskLimits:
    """
    Manages exchange-specific risk limits and constraints.

    Responsibilities:
    - Enforce exchange-specific order size limits
    - Track rate limits to prevent rejections
    - Validate orders against exchange rules
    - Adapt limits based on exchange status
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the exchange risk limits manager.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}

        # Load exchange limit configurations
        self.exchange_limits = {}
        self._load_exchange_limits()

        # Rate limit tracking
        self.rate_limit_windows = {}

        # Last exchange status checks
        self.last_status_check = {}

        # Instrument-specific limits
        self.instrument_limits = {}

        # Data directory for persisting limits
        self.data_dir = Path(self.config.get("data_dir", "data/exchange_limits"))
        self.data_dir.mkdir(parents=True, exist_ok=True)

        logger.info("ExchangeRiskLimits initialized with {} exchanges".format(len(self.exchange_limits)))

    def _load_exchange_limits(self) -> None:
        """Load exchange limit configurations from config"""
        exchange_configs = self.config.get("exchanges", {})

        for exchange_id, config in exchange_configs.items():
            self.exchange_limits[exchange_id] = {
                # General limits
                "enabled": config.get("enabled", True),
                "max_order_size": config.get("max_order_size", float("inf")),
                "min_order_size": config.get("min_order_size", 0.0),
                "max_notional_value": config.get("max_notional_value", float("inf")),
                "min_notional_value": config.get("min_notional_value", 0.0),
                "max_leverage": config.get("max_leverage", float("inf")),
                "price_precision": config.get("price_precision", {}),
                "quantity_precision": config.get("quantity_precision", {}),

                # Rate limits
                "rate_limits": config.get("rate_limits", []),

                # Order types supported
                "supported_order_types": config.get("supported_order_types", []),

                # Time in force options
                "supported_time_in_force": config.get("supported_time_in_force", []),

                # Trading hours
                "trading_hours": config.get("trading_hours", {}),

                # Status indicator
                "status": "online"
            }

            # Initialize rate limit tracking
            self.rate_limit_windows[exchange_id] = {}
            for limit in config.get("rate_limits", []):
                limit_type = limit.get("type", "orders")
                window_seconds = limit.get("window_seconds", 60)

                if limit_type not in self.rate_limit_windows[exchange_id]:
                    self.rate_limit_windows[exchange_id][limit_type] = []

                self.rate_limit_windows[exchange_id][limit_type].append({
                    "window_seconds": window_seconds,
                    "max_count": limit.get("max_count", 100),
                    "timestamps": []
                })

            # Initialize status check
            self.last_status_check[exchange_id] = datetime.now() - timedelta(hours=24)  # Force initial check

    def update_exchange_status(self, exchange_id: str, status: str) -> None:
        """
        Update exchange status.

        Args:
            exchange_id: Exchange ID
            status: New status ('online', 'limited', 'offline')
        """
        if exchange_id not in self.exchange_limits:
            logger.warning(f"Attempted to update status for unknown exchange: {exchange_id}")
            return

        old_status = self.exchange_limits[exchange_id]["status"]
        self.exchange_limits[exchange_id]["status"] = status

        # Log status change
        if old_status != status:
            logger.warning(f"Exchange {exchange_id} status changed: {old_status} -> {status}")

    def get_exchange_status(self, exchange_id: str) -> str:
        """
        Get current exchange status.

        Args:
            exchange_id: Exchange ID

        Returns:
            Exchange status ('online', 'limited', 'offline')
        """
        if exchange_id not in self.exchange_limits:
            logger.warning(f"Requested status for unknown exchange: {exchange_id}")
            return "unknown"

        return self.exchange_limits[exchange_id]["status"]

    def is_exchange_available(self, exchange_id: str) -> bool:
        """
        Check if exchange is available for trading.

        Args:
            exchange_id: Exchange ID

        Returns:
            True if exchange is available, False otherwise
        """
        if exchange_id not in self.exchange_limits:
            return False

        return self.exchange_limits[exchange_id]["status"] in ["online", "limited"]

    def check_order_size_limits(self, exchange_id: str, symbol: str, size: float, price: float) -> Tuple[bool, str]:
        """
        Check if order size meets exchange limits.

        Args:
            exchange_id: Exchange ID
            symbol: Trading symbol
            size: Order size
            price: Order price

        Returns:
            Tuple of (is_valid, reason)
        """
        if exchange_id not in self.exchange_limits:
            return False, f"Unknown exchange: {exchange_id}"

        limits = self.exchange_limits[exchange_id]

        # Check if exchange is enabled
        if not limits["enabled"]:
            return False, f"Exchange {exchange_id} is disabled"

        # Check if exchange is online
        if limits["status"] == "offline":
            return False, f"Exchange {exchange_id} is offline"

        # Check minimum order size
        min_size = limits["min_order_size"]

        # Check for symbol-specific minimum size
        if symbol in self.instrument_limits.get(exchange_id, {}):
            symbol_limits = self.instrument_limits[exchange_id][symbol]
            if "min_order_size" in symbol_limits:
                min_size = symbol_limits["min_order_size"]

        if size < min_size:
            return False, f"Order size {size} below minimum {min_size}"

        # Check maximum order size
        max_size = limits["max_order_size"]

        # Check for symbol-specific maximum size
        if symbol in self.instrument_limits.get(exchange_id, {}):
            symbol_limits = self.instrument_limits[exchange_id][symbol]
            if "max_order_size" in symbol_limits:
                max_size = symbol_limits["max_order_size"]

        if size > max_size:
            return False, f"Order size {size} exceeds maximum {max_size}"

        # Check notional value
        notional_value = size * price
        min_notional = limits["min_notional_value"]
        max_notional = limits["max_notional_value"]

        # Check for symbol-specific notional limits
        if symbol in self.instrument_limits.get(exchange_id, {}):
            symbol_limits = self.instrument_limits[exchange_id][symbol]
            if "min_notional_value" in symbol_limits:
                min_notional = symbol_limits["min_notional_value"]
            if "max_notional_value" in symbol_limits:
                max_notional = symbol_limits["max_notional_value"]

        if notional_value < min_notional:
            return False, f"Order value {notional_value} below minimum {min_notional}"

        if notional_value > max_notional:
            return False, f"Order value {notional_value} exceeds maximum {max_notional}"

        # All checks passed
        return True, ""

    def check_price_precision(self, exchange_id: str, symbol: str, price: float) -> Tuple[bool, float]:
        """
        Check and adjust price precision for exchange.

        Args:
            exchange_id: Exchange ID
            symbol: Trading symbol
            price: Order price

        Returns:
            Tuple of (is_valid, adjusted_price)
        """
        if exchange_id not in self.exchange_limits:
            return False, price

        limits = self.exchange_limits[exchange_id]

        # Get price precision
        precision = limits["price_precision"].get(symbol)

        # Default precision if not specified
        if precision is None:
            default_precision = limits.get("default_price_precision", 8)
            precision = default_precision

        # Adjust price to correct precision
        adjusted_price = round(price, precision)

        return True, adjusted_price

    def check_quantity_precision(self, exchange_id: str, symbol: str, quantity: float) -> Tuple[bool, float]:
        """
        Check and adjust quantity precision for exchange.

        Args:
            exchange_id: Exchange ID
            symbol: Trading symbol
            quantity: Order quantity

        Returns:
            Tuple of (is_valid, adjusted_quantity)
        """
        if exchange_id not in self.exchange_limits:
            return False, quantity

        limits = self.exchange_limits[exchange_id]

        # Get quantity precision
        precision = limits["quantity_precision"].get(symbol)

        # Default precision if not specified
        if precision is None:
            default_precision = limits.get("default_quantity_precision", 8)
            precision = default_precision

        # Adjust quantity to correct precision
        adjusted_quantity = round(quantity, precision)

        return True, adjusted_quantity

    def check_order_type_supported(self, exchange_id: str, order_type: str) -> bool:
        """
        Check if order type is supported by exchange.

        Args:
            exchange_id: Exchange ID
            order_type: Order type

        Returns:
            True if supported, False otherwise
        """
        if exchange_id not in self.exchange_limits:
            return False

        limits = self.exchange_limits[exchange_id]

        return order_type in limits["supported_order_types"]

    def check_time_in_force_supported(self, exchange_id: str, time_in_force: str) -> bool:
        """
        Check if time in force is supported by exchange.

        Args:
            exchange_id: Exchange ID
            time_in_force: Time in force

        Returns:
            True if supported, False otherwise
        """
        if exchange_id not in self.exchange_limits:
            return False

        limits = self.exchange_limits[exchange_id]

        return time_in_force in limits["supported_time_in_force"]

    def check_rate_limit(self, exchange_id: str, limit_type: str = "orders") -> Tuple[bool, int]:
        """
        Check if rate limit allows new operation.

        Args:
            exchange_id: Exchange ID
            limit_type: Rate limit type

        Returns:
            Tuple of (is_allowed, wait_seconds)
        """
        if exchange_id not in self.exchange_limits:
            return False, 0

        if exchange_id not in self.rate_limit_windows:
            return True, 0

        if limit_type not in self.rate_limit_windows[exchange_id]:
            return True, 0

        now = time.time()
        limits_exceeded = False
        max_wait_seconds = 0

        for window in self.rate_limit_windows[exchange_id][limit_type]:
            window_seconds = window["window_seconds"]
            max_count = window["max_count"]

            # Remove timestamps outside the window
            window["timestamps"] = [ts for ts in window["timestamps"] if now - ts < window_seconds]

            # Check if we're at the limit
            if len(window["timestamps"]) >= max_count:
                limits_exceeded = True
                oldest_timestamp = min(window["timestamps"]) if window["timestamps"] else now
                wait_seconds = int(oldest_timestamp + window_seconds - now) + 1
                max_wait_seconds = max(max_wait_seconds, wait_seconds)

        if limits_exceeded:
            logger.warning(f"Rate limit exceeded for {exchange_id} ({limit_type}), wait {max_wait_seconds}s")
            return False, max_wait_seconds

        return True, 0

    def record_request(self, exchange_id: str, limit_type: str = "orders") -> None:
        """
        Record a request for rate limiting.

        Args:
            exchange_id: Exchange ID
            limit_type: Rate limit type
        """
        if exchange_id not in self.rate_limit_windows:
            return

        if limit_type not in self.rate_limit_windows[exchange_id]:
            return

        now = time.time()

        for window in self.rate_limit_windows[exchange_id][limit_type]:
            window["timestamps"].append(now)

    def update_instrument_limits(self, exchange_id: str, symbol: str, limits: Dict[str, Any]) -> None:
        """
        Update instrument-specific limits.

        Args:
            exchange_id: Exchange ID
            symbol: Trading symbol
            limits: Dictionary of limit values
        """
        if exchange_id not in self.instrument_limits:
            self.instrument_limits[exchange_id] = {}

        if symbol not in self.instrument_limits[exchange_id]:
            self.instrument_limits[exchange_id][symbol] = {}

        self.instrument_limits[exchange_id][symbol].update(limits)

        logger.debug(f"Updated instrument limits for {exchange_id}:{symbol} - {limits}")

    def load_limits_from_exchange(self, exchange_id: str, exchange_gateway) -> bool:
        """
        Load limits directly from exchange API.

        Args:
            exchange_id: Exchange ID
            exchange_gateway: Exchange gateway

        Returns:
            True if successful, False otherwise
        """
        try:
            # Get exchange info from gateway
            exchange_info = exchange_gateway.get_exchange_info()

            if not exchange_info:
                logger.warning(f"Failed to load exchange info for {exchange_id}")
                return False

            # Update exchange status
            status = exchange_info.get("status", "online")
            self.update_exchange_status(exchange_id, status)

            # Update rate limits
            rate_limits = exchange_info.get("rate_limits", [])
            if rate_limits:
                self.exchange_limits[exchange_id]["rate_limits"] = rate_limits

                # Reinitialize rate limit windows
                self.rate_limit_windows[exchange_id] = {}
                for limit in rate_limits:
                    limit_type = limit.get("type", "orders")
                    window_seconds = limit.get("window_seconds", 60)

                    if limit_type not in self.rate_limit_windows[exchange_id]:
                        self.rate_limit_windows[exchange_id][limit_type] = []

                    self.rate_limit_windows[exchange_id][limit_type].append({
                        "window_seconds": window_seconds,
                        "max_count": limit.get("max_count", 100),
                        "timestamps": []
                    })

            # Update trading hours
            trading_hours = exchange_info.get("trading_hours", {})
            if trading_hours:
                self.exchange_limits[exchange_id]["trading_hours"] = trading_hours

            # Update symbol-specific limits
            symbols = exchange_info.get("symbols", [])
            for symbol_info in symbols:
                symbol = symbol_info.get("symbol")
                if not symbol:
                    continue

                symbol_limits = {}

                # Extract limit fields
                if "min_order_size" in symbol_info:
                    symbol_limits["min_order_size"] = symbol_info["min_order_size"]

                if "max_order_size" in symbol_info:
                    symbol_limits["max_order_size"] = symbol_info["max_order_size"]

                if "min_notional_value" in symbol_info:
                    symbol_limits["min_notional_value"] = symbol_info["min_notional_value"]

                if "max_notional_value" in symbol_info:
                    symbol_limits["max_notional_value"] = symbol_info["max_notional_value"]

                if "price_precision" in symbol_info:
                    symbol_limits["price_precision"] = symbol_info["price_precision"]
                    # Also update in exchange limits
                    self.exchange_limits[exchange_id]["price_precision"][symbol] = symbol_info["price_precision"]

                if "quantity_precision" in symbol_info:
                    symbol_limits["quantity_precision"] = symbol_info["quantity_precision"]
                    # Also update in exchange limits
                    self.exchange_limits[exchange_id]["quantity_precision"][symbol] = symbol_info["quantity_precision"]

                # Update instrument limits
                if symbol_limits:
                    self.update_instrument_limits(exchange_id, symbol, symbol_limits)

            # Save updated limits
            self._save_limits_to_file(exchange_id)

            self.last_status_check[exchange_id] = datetime.now()
            logger.info(f"Successfully loaded exchange limits for {exchange_id}")

            return True

        except Exception as e:
            logger.error(f"Error loading exchange limits for {exchange_id}: {str(e)}", exc_info=True)
            return False

    def _save_limits_to_file(self, exchange_id: str) -> None:
        """
        Save exchange limits to file.

        Args:
            exchange_id: Exchange ID
        """
        try:
            file_path = self.data_dir / f"{exchange_id}_limits.json"

            if not file_path.exists():
                logger.debug(f"No saved limits file for {exchange_id}")
                return False

            with open(file_path, 'r') as f:
                data = json.load(f)

            # Load exchange limits
            if "exchange_limits" in data:
                self.exchange_limits[exchange_id] = data["exchange_limits"]

            # Load instrument limits
            if "instrument_limits" in data:
                if exchange_id not in self.instrument_limits:
                    self.instrument_limits[exchange_id] = {}
                self.instrument_limits[exchange_id].update(data["instrument_limits"])

            # Re-initialize rate limit tracking
            self.rate_limit_windows[exchange_id] = {}
            for limit in self.exchange_limits[exchange_id].get("rate_limits", []):
                limit_type = limit.get("type", "orders")
                window_seconds = limit.get("window_seconds", 60)

                if limit_type not in self.rate_limit_windows[exchange_id]:
                    self.rate_limit_windows[exchange_id][limit_type] = []

                self.rate_limit_windows[exchange_id][limit_type].append({
                    "window_seconds": window_seconds,
                    "max_count": limit.get("max_count", 100),
                    "timestamps": []
                })

            logger.info(f"Loaded exchange limits for {exchange_id} from {file_path}")
            return True

        except Exception as e:
            logger.error(f"Error loading exchange limits from file for {exchange_id}: {str(e)}")
            return False

    def check_trading_hours(self, exchange_id: str, symbol: str) -> bool:
        """
        Check if current time is within trading hours for symbol.

        Args:
            exchange_id: Exchange ID
            symbol: Trading symbol

        Returns:
            True if trading is allowed, False otherwise
        """
        if exchange_id not in self.exchange_limits:
            return False

        limits = self.exchange_limits[exchange_id]

        # Check if trading hours are defined
        if "trading_hours" not in limits or not limits["trading_hours"]:
            return True  # Default to allowed if not specified

        # Get trading hours for symbol
        trading_hours = limits["trading_hours"].get(symbol)

        # If not symbol-specific, check for default
        if not trading_hours:
            trading_hours = limits["trading_hours"].get("default")

        # If no trading hours defined, assume allowed
        if not trading_hours:
            return True

        # Check if 24/7 trading
        if trading_hours.get("24/7", False):
            return True

        # Get current time in UTC
        now = datetime.utcnow()
        current_day = now.strftime("%A").lower()  # e.g., "monday"
        current_time = now.time()

        # Check day first
        if current_day not in trading_hours:
            return False

        # Check time windows for current day
        for window in trading_hours[current_day]:
            open_time_str = window.get("open")
            close_time_str = window.get("close")

            if not open_time_str or not close_time_str:
                continue

            # Parse time strings
            try:
                open_time = datetime.strptime(open_time_str, "%H:%M:%S").time()
                close_time = datetime.strptime(close_time_str, "%H:%M:%S").time()

                # Check if current time is within window
                if open_time <= current_time <= close_time:
                    return True

            except ValueError:
                logger.warning(f"Invalid time format in trading hours for {exchange_id}:{symbol}")

        # No matching window found
        return False

    def check_leverage_limit(self, exchange_id: str, symbol: str, leverage: float) -> Tuple[bool, float]:
        """
        Check and adjust leverage against exchange limits.

        Args:
            exchange_id: Exchange ID
            symbol: Trading symbol
            leverage: Requested leverage

        Returns:
            Tuple of (is_valid, adjusted_leverage)
        """
        if exchange_id not in self.exchange_limits:
            return False, leverage

        limits = self.exchange_limits[exchange_id]

        # Get max leverage
        max_leverage = limits["max_leverage"]

        # Check symbol-specific leverage
        if symbol in self.instrument_limits.get(exchange_id, {}):
            symbol_limits = self.instrument_limits[exchange_id][symbol]
            if "max_leverage" in symbol_limits:
                max_leverage = symbol_limits["max_leverage"]

        # Adjust leverage if needed
        if leverage > max_leverage:
            logger.warning(f"Requested leverage {leverage} exceeds maximum {max_leverage} for {exchange_id}:{symbol}")
            return True, max_leverage

        return True, leverage

    def validate_order(self, exchange_id: str, symbol: str, order_type: str,
                      time_in_force: str, size: float, price: float = None,
                      leverage: float = None) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Comprehensive order validation against exchange limits.

        Args:
            exchange_id: Exchange ID
            symbol: Trading symbol
            order_type: Order type
            time_in_force: Time in force
            size: Order size
            price: Order price (optional, required for limit orders)
            leverage: Order leverage (optional)

        Returns:
            Tuple of (is_valid, error_message, adjusted_params)
        """
        adjusted_params = {}

        # Check exchange status
        if not self.is_exchange_available(exchange_id):
            return False, f"Exchange {exchange_id} is not available", adjusted_params

        # Check trading hours
        if not self.check_trading_hours(exchange_id, symbol):
            return False, f"Trading hours closed for {symbol} on {exchange_id}", adjusted_params

        # Check order type support
        if not self.check_order_type_supported(exchange_id, order_type):
            return False, f"Order type {order_type} not supported by {exchange_id}", adjusted_params

        # Check time in force support
        if not self.check_time_in_force_supported(exchange_id, time_in_force):
            return False, f"Time in force {time_in_force} not supported by {exchange_id}", adjusted_params

        # For limit orders, need price
        if order_type == "limit" and price is None:
            return False, "Limit order requires price", adjusted_params

        # Check rate limits
        is_allowed, wait_seconds = self.check_rate_limit(exchange_id, "orders")
        if not is_allowed:
            return False, f"Rate limit exceeded for {exchange_id}, wait {wait_seconds}s", adjusted_params

        # Check price precision
        if price is not None:
            is_valid, adjusted_price = self.check_price_precision(exchange_id, symbol, price)
            if not is_valid:
                return False, "Invalid price precision", adjusted_params

            if adjusted_price != price:
                adjusted_params["price"] = adjusted_price

        # Check quantity precision
        is_valid, adjusted_size = self.check_quantity_precision(exchange_id, symbol, size)
        if not is_valid:
            return False, "Invalid quantity precision", adjusted_params

        if adjusted_size != size:
            adjusted_params["size"] = adjusted_size

        # Check leverage limits
        if leverage is not None:
            is_valid, adjusted_leverage = self.check_leverage_limit(exchange_id, symbol, leverage)
            if not is_valid:
                return False, "Invalid leverage", adjusted_params

            if adjusted_leverage != leverage:
                adjusted_params["leverage"] = adjusted_leverage

        # Check size limits with adjusted size and price
        actual_size = adjusted_params.get("size", size)
        actual_price = adjusted_params.get("price", price) if price is not None else None

        # For market orders without price, can't check notional value
        if order_type == "market" and actual_price is None:
            # Check only size limits, not notional
            is_valid, reason = self.check_order_size_limits(exchange_id, symbol, actual_size, float("inf"))
        else:
            # Check both size and notional limits
            is_valid, reason = self.check_order_size_limits(exchange_id, symbol, actual_size, actual_price)

        if not is_valid:
            return False, reason, adjusted_params

        # Record this request for rate limiting
        self.record_request(exchange_id, "orders")

        return True, "", adjusted_params

    def should_check_exchange_status(self, exchange_id: str) -> bool:
        """
        Determine if exchange status should be checked.

        Args:
            exchange_id: Exchange ID

        Returns:
            True if status should be checked, False otherwise
        """
        if exchange_id not in self.last_status_check:
            return True

        # Get status check interval
        status_check_interval = self.config.get("status_check_interval_minutes", 60)

        # Check if enough time has passed since last check
        time_since_check = datetime.now() - self.last_status_check[exchange_id]

        return time_since_check.total_seconds() / 60 >= status_check_interval

    def get_limits_summary(self, exchange_id: str, symbol: Optional[str] = None) -> Dict[str, Any]:
        """
        Get a summary of limits for an exchange.

        Args:
            exchange_id: Exchange ID
            symbol: Optional specific symbol

        Returns:
            Dictionary with limits summary
        """
        if exchange_id not in self.exchange_limits:
            return {"error": f"Unknown exchange: {exchange_id}"}

        limits = self.exchange_limits[exchange_id]

        summary = {
            "exchange_id": exchange_id,
            "status": limits["status"],
            "enabled": limits["enabled"],
            "general_limits": {
                "max_order_size": limits["max_order_size"],
                "min_order_size": limits["min_order_size"],
                "max_notional_value": limits["max_notional_value"],
                "min_notional_value": limits["min_notional_value"],
                "max_leverage": limits["max_leverage"]
            },
            "supported_order_types": limits["supported_order_types"],
            "supported_time_in_force": limits["supported_time_in_force"],
            "rate_limits": limits["rate_limits"]
        }

        # Add symbol-specific limits if requested
        if symbol and exchange_id in self.instrument_limits and symbol in self.instrument_limits[exchange_id]:
            summary["symbol_limits"] = self.instrument_limits[exchange_id][symbol]

        return summary

    def reset(self) -> None:
        """Reset rate limit tracking but preserve limit configurations."""
        # Reset rate limit tracking
        for exchange_id in self.rate_limit_windows:
            for limit_type in self.rate_limit_windows[exchange_id]:
                for window in self.rate_limit_windows[exchange_id][limit_type]:
                    window["timestamps"] = []

        # Reset status check times
        self.last_status_check = {exchange_id: datetime.now() - timedelta(hours=24) for exchange_id in self.exchange_limits}

        logger.info("Exchange risk limits reset")

            data = {
                "exchange_limits": self.exchange_limits.get(exchange_id, {}),
                "instrument_limits": self.instrument_limits.get(exchange_id, {}),
                "updated_at": datetime.now().isoformat()
            }

            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)

            logger.debug(f"Saved exchange limits for {exchange_id} to {file_path}")

        except Exception as e:
            logger.error(f"Error saving exchange limits for {exchange_id}: {str(e)}")

    def _load_limits_from_file(self, exchange_id: str) -> bool:
        """
        Load exchange limits from file.

        Args:
            exchange_id: Exchange ID

        Returns:
            True if successful, False otherwise
        """
        try:
            file_path = self.data_dir / f"{exchange_id}_limits.json"