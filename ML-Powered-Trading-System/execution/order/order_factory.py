"""
Order Factory Module

This module provides a factory for creating various types of orders,
ensuring consistent order creation throughout the system.
"""

import logging
from typing import Dict, Any, Optional, List, Union
from datetime import datetime

from .order import (
    Order, OrderType, OrderSide, OrderStatus, TimeInForce
)

# Configure logger
logger = logging.getLogger(__name__)

class OrderFactory:
    """
    Factory class for creating different types of trading orders.

    This class provides methods to create various types of orders with
    consistent configuration and validation.
    """

    def __init__(self):
        """Initialize the order factory"""
        self.default_exchange_id = None
        self.default_account = None
        self.default_time_in_force = TimeInForce.GTC
        self.default_tags = []
        self.default_params = {}

    def configure(self,
                 default_exchange_id: Optional[str] = None,
                 default_account: Optional[str] = None,
                 default_time_in_force: Optional[TimeInForce] = None,
                 default_tags: Optional[List[str]] = None,
                 default_params: Optional[Dict[str, Any]] = None) -> None:
        """
        Configure default values for order creation.

        Args:
            default_exchange_id: Default exchange to use
            default_account: Default account to use
            default_time_in_force: Default time-in-force setting
            default_tags: Default tags to apply to orders
            default_params: Default params to include in orders
        """
        if default_exchange_id is not None:
            self.default_exchange_id = default_exchange_id

        if default_account is not None:
            self.default_account = default_account

        if default_time_in_force is not None:
            self.default_time_in_force = default_time_in_force

        if default_tags is not None:
            self.default_tags = default_tags.copy()

        if default_params is not None:
            self.default_params = default_params.copy()

        logger.info(f"OrderFactory configured with defaults: exchange={self.default_exchange_id}, "
                   f"account={self.default_account}, time_in_force={self.default_time_in_force}")

    def _prepare_common_params(self, **kwargs) -> Dict[str, Any]:
        """
        Prepare common parameters for order creation.

        This method applies defaults for exchange, account, time_in_force,
        tags, and params if they are not specified in kwargs.

        Args:
            **kwargs: Order parameters

        Returns:
            Dict with common parameters including defaults
        """
        params = kwargs.copy()

        # Apply defaults if not specified
        if 'exchange_id' not in params and self.default_exchange_id:
            params['exchange_id'] = self.default_exchange_id

        if 'exchange_account' not in params and self.default_account:
            params['exchange_account'] = self.default_account

        if 'time_in_force' not in params and self.default_time_in_force:
            params['time_in_force'] = self.default_time_in_force

        # Merge tags
        if self.default_tags:
            tags = list(params.get('tags', []))
            tags.extend(tag for tag in self.default_tags if tag not in tags)
            params['tags'] = tags

        # Merge params
        if self.default_params:
            order_params = self.default_params.copy()
            order_params.update(params.get('params', {}))
            params['params'] = order_params

        return params

    def create_market_order(
        self,
        symbol: str,
        side: Union[OrderSide, str],
        quantity: float,
        **kwargs
    ) -> Order:
        """
        Create a market order.

        Args:
            symbol: Trading symbol
            side: Order side (buy or sell)
            quantity: Order quantity
            **kwargs: Additional order parameters

        Returns:
            Market order instance
        """
        # Convert string side to enum if needed
        if isinstance(side, str):
            side = OrderSide(side.lower())

        # Prepare common parameters
        params = self._prepare_common_params(**kwargs)

        # Create the order
        order = Order(
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=OrderType.MARKET,
            **params
        )

        logger.info(f"Created market {side.value} order for {symbol}: quantity={quantity}")
        return order

    def create_limit_order(
        self,
        symbol: str,
        side: Union[OrderSide, str],
        quantity: float,
        price: float,
        **kwargs
    ) -> Order:
        """
        Create a limit order.

        Args:
            symbol: Trading symbol
            side: Order side (buy or sell)
            quantity: Order quantity
            price: Limit price
            **kwargs: Additional order parameters

        Returns:
            Limit order instance
        """
        # Convert string side to enum if needed
        if isinstance(side, str):
            side = OrderSide(side.lower())

        # Prepare common parameters
        params = self._prepare_common_params(**kwargs)

        # Create the order
        order = Order(
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=OrderType.LIMIT,
            price=price,
            **params
        )

        logger.info(f"Created limit {side.value} order for {symbol}: "
                   f"quantity={quantity}, price={price}")
        return order

    def create_stop_order(
        self,
        symbol: str,
        side: Union[OrderSide, str],
        quantity: float,
        stop_price: float,
        **kwargs
    ) -> Order:
        """
        Create a stop (market) order.

        Args:
            symbol: Trading symbol
            side: Order side (buy or sell)
            quantity: Order quantity
            stop_price: Stop trigger price
            **kwargs: Additional order parameters

        Returns:
            Stop order instance
        """
        # Convert string side to enum if needed
        if isinstance(side, str):
            side = OrderSide(side.lower())

        # Prepare common parameters
        params = self._prepare_common_params(**kwargs)

        # Create the order
        order = Order(
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=OrderType.STOP,
            stop_price=stop_price,
            **params
        )

        logger.info(f"Created stop {side.value} order for {symbol}: "
                   f"quantity={quantity}, stop_price={stop_price}")
        return order

    def create_stop_limit_order(
        self,
        symbol: str,
        side: Union[OrderSide, str],
        quantity: float,
        stop_price: float,
        limit_price: float,
        **kwargs
    ) -> Order:
        """
        Create a stop-limit order.

        Args:
            symbol: Trading symbol
            side: Order side (buy or sell)
            quantity: Order quantity
            stop_price: Stop trigger price
            limit_price: Limit price after stop is triggered
            **kwargs: Additional order parameters

        Returns:
            Stop-limit order instance
        """
        # Convert string side to enum if needed
        if isinstance(side, str):
            side = OrderSide(side.lower())

        # Prepare common parameters
        params = self._prepare_common_params(**kwargs)

        # Create the order
        order = Order(
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=OrderType.STOP_LIMIT,
            price=limit_price,
            stop_price=stop_price,
            **params
        )

        logger.info(f"Created stop-limit {side.value} order for {symbol}: "
                   f"quantity={quantity}, stop_price={stop_price}, limit_price={limit_price}")
        return order

    def create_trailing_stop_order(
        self,
        symbol: str,
        side: Union[OrderSide, str],
        quantity: float,
        trail_amount: Optional[float] = None,
        trail_percent: Optional[float] = None,
        **kwargs
    ) -> Order:
        """
        Create a trailing stop order.

        Args:
            symbol: Trading symbol
            side: Order side (buy or sell)
            quantity: Order quantity
            trail_amount: Fixed amount for trailing (absolute price difference)
            trail_percent: Percentage amount for trailing (relative to price)
            **kwargs: Additional order parameters

        Returns:
            Trailing stop order instance
        """
        # Convert string side to enum if needed
        if isinstance(side, str):
            side = OrderSide(side.lower())

        # Ensure at least one trailing parameter is specified
        if trail_amount is None and trail_percent is None:
            raise ValueError("Either trail_amount or trail_percent must be specified")

        # Prepare common parameters
        params = self._prepare_common_params(**kwargs)

        # Add trailing parameters to the params dictionary
        order_params = params.get('params', {})
        if trail_amount is not None:
            order_params['trail_amount'] = trail_amount
        if trail_percent is not None:
            order_params['trail_percent'] = trail_percent
        params['params'] = order_params

        # Create the order
        order = Order(
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=OrderType.TRAILING_STOP,
            **params
        )

        logger.info(f"Created trailing stop {side.value} order for {symbol}: "
                   f"quantity={quantity}, {'trail_amount=' + str(trail_amount) if trail_amount is not None else 'trail_percent=' + str(trail_percent) + '%'}")
        return order

    def create_order(self,
                    symbol: str,
                    side: Union[OrderSide, str],
                    quantity: float,
                    order_type: Union[OrderType, str],
                    **kwargs) -> Order:
        """
        Generic method to create any type of order.

        Args:
            symbol: Trading symbol
            side: Order side (buy or sell)
            quantity: Order quantity
            order_type: Type of order to create
            **kwargs: Additional order parameters

        Returns:
            The created order

        Raises:
            ValueError: If invalid order type or missing required parameters
        """
        # Convert string values to enums if needed
        if isinstance(side, str):
            side = OrderSide(side.lower())

        if isinstance(order_type, str):
            order_type = OrderType(order_type.lower())

        # Route to appropriate creation method based on order type
        if order_type == OrderType.MARKET:
            return self.create_market_order(symbol, side, quantity, **kwargs)
        elif order_type == OrderType.LIMIT:
            if 'price' not in kwargs:
                raise ValueError("Price required for limit orders")
            return self.create_limit_order(symbol, side, quantity, kwargs.pop('price'), **kwargs)
        elif order_type == OrderType.STOP:
            if 'stop_price' not in kwargs:
                raise ValueError("Stop price required for stop orders")
            return self.create_stop_order(symbol, side, quantity, kwargs.pop('stop_price'), **kwargs)
        elif order_type == OrderType.STOP_LIMIT:
            if 'stop_price' not in kwargs or 'price' not in kwargs:
                raise ValueError("Stop price and limit price required for stop-limit orders")
            return self.create_stop_limit_order(
                symbol, side, quantity,
                kwargs.pop('stop_price'),
                kwargs.pop('price'),
                **kwargs
            )
        elif order_type == OrderType.TRAILING_STOP:
            if 'trail_amount' not in kwargs and 'trail_percent' not in kwargs:
                raise ValueError("Either trail_amount or trail_percent required for trailing stop orders")
            return self.create_trailing_stop_order(
                symbol, side, quantity,
                kwargs.pop('trail_amount', None),
                kwargs.pop('trail_percent', None),
                **kwargs
            )
        else:
            # For any other order type, create a generic order
            params = self._prepare_common_params(**kwargs)
            return Order(
                symbol=symbol,
                side=side,
                quantity=quantity,
                order_type=order_type,
                **params
            )

    def create_bulk_orders(self, orders_data: List[Dict[str, Any]]) -> List[Order]:
        """
        Create multiple orders at once.

        Args:
            orders_data: List of dictionaries with order parameters

        Returns:
            List of created orders
        """
        orders = []

        for order_data in orders_data:
            try:
                # Extract required parameters
                symbol = order_data.pop('symbol')
                side = order_data.pop('side')
                quantity = order_data.pop('quantity')
                order_type = order_data.pop('order_type')

                # Create order
                order = self.create_order(symbol, side, quantity, order_type, **order_data)
                orders.append(order)

            except (KeyError, ValueError) as e:
                logger.error(f"Error creating order from data {order_data}: {str(e)}")
                # Skip this order but continue with others
                continue

        return orders