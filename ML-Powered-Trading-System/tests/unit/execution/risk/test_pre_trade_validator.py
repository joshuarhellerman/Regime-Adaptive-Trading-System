import unittest
from unittest.mock import Mock, patch, MagicMock
import pytest
from datetime import datetime

from execution.exchange.risk.pre_trade_validator import (
    PreTradeValidator, RiskCheckLevel, RiskCheckResult
)
from execution.order.order import Order, OrderSide, OrderType, TimeInForce


class TestPreTradeValidator(unittest.TestCase):
    """Test suite for PreTradeValidator."""

    def setUp(self):
        """Set up test fixtures before each test."""
        self.config = {
            "max_order_size": 1000,
            "max_order_value": 50000,
            "max_position_size": 5000,
            "max_position_value": 250000,
            "max_concentration": 0.2,
            "max_overnight_exposure": 0.5,
            "max_order_count_per_minute": 20,
            "max_daily_drawdown": 0.03,
            "price_deviation_threshold": 0.05,
            "min_order_size": 0.001,
            "max_leverage": 5,
            "max_open_orders": 10,
            "max_total_exposure": 3.0,
            "instrument_limits": {
                "BTC-USD": {
                    "max_order_size": 10,
                    "max_position_size": 50,
                    "price_deviation_threshold": 0.07
                },
                "AAPL": {
                    "max_order_size": 500,
                    "max_concentration": 0.15
                }
            },
            "exchange_limits": {
                "binance": {
                    "max_order_size": 200
                },
                "kraken": {
                    "allowed": False
                }
            },
            "strategy_limits": {
                "momentum": {
                    "max_position_size": 3000,
                    "max_equity_percent": 0.1
                },
                "market_making": {
                    "max_position_size": 2000,
                    "max_order_count_per_minute": 50
                }
            },
            "regime_adjustments": {
                "high_volatility": {
                    "max_order_size_multiplier": 0.5,
                    "price_deviation_threshold_multiplier": 2.0
                }
            }
        }

        self.validator = PreTradeValidator(self.config)

        # Create a basic test order
        self.base_order = Order(
            order_id="test-order-001",
            symbol="SPY",
            quantity=100,
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            limit_price=400.0,
            time_in_force=TimeInForce.DAY,
            exchange_id="test_exchange",
            params={"strategy": "momentum", "equity": 100000}
        )

    def test_init(self):
        """Test validator initialization."""
        validator = PreTradeValidator(self.config)

        # Check if risk limits are properly initialized
        self.assertEqual(validator.risk_limits["max_order_size"], 1000)
        self.assertEqual(validator.risk_limits["max_leverage"], 5)

        # Check if instrument limits are properly initialized
        self.assertEqual(validator.instrument_limits["BTC-USD"]["max_order_size"], 10)
        self.assertEqual(validator.instrument_limits["AAPL"]["max_concentration"], 0.15)

        # Check if exchange limits are properly initialized
        self.assertEqual(validator.exchange_limits["binance"]["max_order_size"], 200)
        self.assertEqual(validator.exchange_limits["kraken"]["allowed"], False)

    def test_validate_order_integration(self):
        """Test the full validate_order method with mocked dependencies."""
        # Mock dependencies
        position_manager = Mock()
        position_manager.get_position.return_value = 50
        position_manager.get_total_position_value.return_value = 500000
        position_manager.get_position_value.return_value = 20000
        position_manager.get_market_price.return_value = 400.0
        position_manager.get_all_positions.return_value = {"SPY": Mock(open_orders=[])}

        market_data_service = Mock()
        market_data_service.get_last_price.return_value = 400.0
        market_data_service.is_market_open.return_value = True
        market_data_service.get_volatility.return_value = 0.02
        market_data_service.get_liquidity.return_value = 5000000

        account_manager = Mock()
        account_manager.get_available_capital.return_value = 200000
        account_manager.get_equity.return_value = 100000
        account_manager.get_current_leverage.return_value = 2.0
        account_manager.check_margin_requirement.return_value = {"passed": True}

        # Test a valid order
        results = self.validator.validate_order(
            self.base_order, position_manager, market_data_service, account_manager
        )

        # All checks should pass or be informational
        for result in results:
            self.assertNotEqual(result.level, RiskCheckLevel.ERROR)

    def test_check_order_parameters(self):
        """Test the order parameters validation."""
        # Valid order should pass
        result = self.validator._check_order_parameters(self.base_order)
        self.assertEqual(result.level, RiskCheckLevel.INFO)

        # Missing symbol
        order = Order(
            order_id="test-order-002",
            symbol="",
            quantity=100,
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            limit_price=400.0
        )
        result = self.validator._check_order_parameters(order)
        self.assertEqual(result.level, RiskCheckLevel.ERROR)
        self.assertIn("Missing symbol", result.message)

        # Missing quantity
        order = Order(
            order_id="test-order-003",
            symbol="SPY",
            quantity=0,
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            limit_price=400.0
        )
        result = self.validator._check_order_parameters(order)
        self.assertEqual(result.level, RiskCheckLevel.ERROR)
        self.assertIn("Missing or zero quantity", result.message)

        # Negative quantity
        order = Order(
            order_id="test-order-004",
            symbol="SPY",
            quantity=-10,
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            limit_price=400.0
        )
        result = self.validator._check_order_parameters(order)
        self.assertEqual(result.level, RiskCheckLevel.ERROR)
        self.assertIn("Negative quantity", result.message)

        # Limit order missing price
        order = Order(
            order_id="test-order-005",
            symbol="SPY",
            quantity=100,
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            limit_price=None
        )
        result = self.validator._check_order_parameters(order)
        self.assertEqual(result.level, RiskCheckLevel.ERROR)
        self.assertIn("Limit order missing price", result.message)

        # Stop order missing stop price
        order = Order(
            order_id="test-order-006",
            symbol="SPY",
            quantity=100,
            side=OrderSide.BUY,
            order_type=OrderType.STOP,
            stop_price=None
        )
        result = self.validator._check_order_parameters(order)
        self.assertEqual(result.level, RiskCheckLevel.ERROR)
        self.assertIn("order missing stop price", result.message)

        # Order size below minimum
        self.validator.risk_limits["min_order_size"] = 200
        result = self.validator._check_order_parameters(self.base_order)
        self.assertEqual(result.level, RiskCheckLevel.ERROR)
        self.assertIn("below minimum", result.message)

    def test_check_order_size_limits(self):
        """Test the order size limits validation."""
        # Valid order should pass
        result = self.validator._check_order_size_limits(self.base_order)
        self.assertEqual(result.level, RiskCheckLevel.INFO)

        # Order size exceeds max
        order = Order(
            order_id="test-order-007",
            symbol="SPY",
            quantity=2000,  # Above max_order_size of 1000
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            limit_price=400.0
        )
        result = self.validator._check_order_size_limits(order)
        self.assertEqual(result.level, RiskCheckLevel.ERROR)
        self.assertIn("exceeds maximum", result.message)

        # Order value check with mock market data service
        market_data_service = Mock()
        market_data_service.get_last_price.return_value = 500.0

        # Order value within limits
        result = self.validator._check_order_size_limits(self.base_order, market_data_service)
        self.assertEqual(result.level, RiskCheckLevel.INFO)

        # Order value exceeds max
        order = Order(
            order_id="test-order-008",
            symbol="SPY",
            quantity=200,
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            limit_price=300.0
        )
        market_data_service.get_last_price.return_value = 300.0
        self.validator.risk_limits["max_order_value"] = 50000
        result = self.validator._check_order_size_limits(order, market_data_service)
        self.assertEqual(result.level, RiskCheckLevel.INFO)  # 200 * 300 = 60000, below 50000

        order.quantity = 500
        result = self.validator._check_order_size_limits(order, market_data_service)
        self.assertEqual(result.level, RiskCheckLevel.ERROR)  # 500 * 300 = 150000, exceeds 50000

        # Test instrument-specific limits
        order = Order(
            order_id="test-order-009",
            symbol="BTC-USD",
            quantity=5,
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            limit_price=50000.0
        )
        result = self.validator._check_order_size_limits(order)
        self.assertEqual(result.level, RiskCheckLevel.INFO)  # 5 is below 10

        order.quantity = 15
        result = self.validator._check_order_size_limits(order)
        self.assertEqual(result.level, RiskCheckLevel.ERROR)  # 15 exceeds instrument limit of 10

    def test_check_position_limits(self):
        """Test the position limits validation."""
        # Mock position manager
        position_manager = Mock()
        position_manager.get_position.return_value = 50  # Current position

        # Buy order that keeps position within limits
        result = self.validator._check_position_limits(self.base_order, position_manager)
        self.assertEqual(result.level, RiskCheckLevel.INFO)

        # Buy order that would exceed position limits
        order = Order(
            order_id="test-order-010",
            symbol="SPY",
            quantity=5000,
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            limit_price=400.0
        )
        result = self.validator._check_position_limits(order, position_manager)
        self.assertEqual(result.level, RiskCheckLevel.ERROR)
        self.assertIn("would exceed maximum", result.message)

        # Sell order that keeps position within limits
        order.side = OrderSide.SELL
        result = self.validator._check_position_limits(order, position_manager)
        self.assertEqual(result.level, RiskCheckLevel.INFO)

        # Test instrument-specific position limits
        order = Order(
            order_id="test-order-011",
            symbol="BTC-USD",
            quantity=40,
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            limit_price=50000.0
        )
        position_manager.get_position.return_value = 20
        result = self.validator._check_position_limits(order, position_manager)
        self.assertEqual(result.level, RiskCheckLevel.ERROR)  # 20 + 40 = 60, exceeds 50

    def test_check_concentration_limits(self):
        """Test the concentration limits validation."""
        # Mock position manager
        position_manager = Mock()
        position_manager.get_total_position_value.return_value = 500000
        position_manager.get_position_value.return_value = 20000
        position_manager.get_market_price.return_value = 400.0

        # Order that keeps concentration within limits
        result = self.validator._check_concentration_limits(self.base_order, position_manager)
        self.assertEqual(result.level, RiskCheckLevel.INFO)

        # Order that would exceed concentration limits
        order = Order(
            order_id="test-order-012",
            symbol="SPY",
            quantity=2000,
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            limit_price=400.0
        )
        result = self.validator._check_concentration_limits(order, position_manager)
        self.assertEqual(result.level, RiskCheckLevel.ERROR)
        self.assertIn("would exceed maximum", result.message)

        # Test instrument-specific concentration limits
        order = Order(
            order_id="test-order-013",
            symbol="AAPL",
            quantity=1000,
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            limit_price=150.0
        )
        position_manager.get_position_value.return_value = 10000
        result = self.validator._check_concentration_limits(order, position_manager)
        self.assertEqual(result.level, RiskCheckLevel.ERROR)  # (10k + 150k)/500k = 0.32, exceeds 0.15

    def test_check_price_deviation(self):
        """Test the price deviation validation."""
        # Mock market data service
        market_data_service = Mock()
        market_data_service.get_last_price.return_value = 400.0

        # Order with price within deviation limits
        order = Order(
            order_id="test-order-014",
            symbol="SPY",
            quantity=100,
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            limit_price=410.0  # 2.5% deviation
        )
        result = self.validator._check_price_deviation(order, market_data_service)
        self.assertEqual(result.level, RiskCheckLevel.INFO)

        # Order with price exceeding deviation limits
        order.limit_price = 450.0  # 12.5% deviation
        result = self.validator._check_price_deviation(order, market_data_service)
        self.assertEqual(result.level, RiskCheckLevel.WARNING)
        self.assertIn("exceeds threshold", result.message)

        # Order with severe price deviation
        order.limit_price = 500.0  # 25% deviation
        result = self.validator._check_price_deviation(order, market_data_service)
        self.assertEqual(result.level, RiskCheckLevel.ERROR)

        # Test instrument-specific deviation thresholds
        order = Order(
            order_id="test-order-015",
            symbol="BTC-USD",
            quantity=1,
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            limit_price=53500.0  # 7% deviation
        )
        market_data_service.get_last_price.return_value = 50000.0
        result = self.validator._check_price_deviation(order, market_data_service)
        self.assertEqual(result.level, RiskCheckLevel.INFO)  # Within 7% threshold for BTC-USD

    def test_check_market_hours(self):
        """Test the market hours validation."""
        # Mock market data service
        market_data_service = Mock()
        market_data_service.is_market_open.return_value = True

        # Market is open
        result = self.validator._check_market_hours(self.base_order, market_data_service)
        self.assertEqual(result.level, RiskCheckLevel.INFO)
        self.assertIn("open for trading", result.message)

        # Market is closed
        market_data_service.is_market_open.return_value = False
        result = self.validator._check_market_hours(self.base_order, market_data_service)
        self.assertEqual(result.level, RiskCheckLevel.ERROR)
        self.assertIn("Market is closed", result.message)

        # Market is closed but after-hours trading is allowed
        self.validator.config["allow_after_hours"] = True
        result = self.validator._check_market_hours(self.base_order, market_data_service)
        self.assertEqual(result.level, RiskCheckLevel.WARNING)
        self.assertIn("after-hours trading is allowed", result.message)

    def test_check_order_frequency(self):
        """Test the order frequency validation."""
        # No recent orders
        result = self.validator._check_order_frequency(self.base_order)
        self.assertEqual(result.level, RiskCheckLevel.INFO)

        # Add some recent orders
        self.validator._record_order_for_rate_limiting(self.base_order)
        result = self.validator._check_order_frequency(self.base_order)
        self.assertEqual(result.level, RiskCheckLevel.INFO)

        # Add more orders to reach limit
        for _ in range(20):
            self.validator._record_order_for_rate_limiting(self.base_order)

        result = self.validator._check_order_frequency(self.base_order)
        self.assertEqual(result.level, RiskCheckLevel.ERROR)
        self.assertIn("exceeds limit", result.message)

    def test_check_instrument_specific_limits(self):
        """Test the instrument-specific limits validation."""
        # Regular instrument
        result = self.validator._check_instrument_specific_limits(self.base_order)
        self.assertEqual(result.level, RiskCheckLevel.INFO)

        # Instrument with specific limits
        order = Order(
            order_id="test-order-016",
            symbol="BTC-USD",
            quantity=5,
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            limit_price=50000.0
        )
        result = self.validator._check_instrument_specific_limits(order)
        self.assertEqual(result.level, RiskCheckLevel.INFO)

        # Instrument that is not allowed
        self.validator.instrument_limits["BTC-USD"]["allowed"] = False
        result = self.validator._check_instrument_specific_limits(order)
        self.assertEqual(result.level, RiskCheckLevel.ERROR)
        self.assertIn("not allowed", result.message)

    def test_check_exchange_specific_limits(self):
        """Test the exchange-specific limits validation."""
        # Regular exchange
        result = self.validator._check_exchange_specific_limits(self.base_order)
        self.assertEqual(result.level, RiskCheckLevel.INFO)

        # Exchange with specific limits
        order = Order(
            order_id="test-order-017",
            symbol="BTC-USD",
            quantity=100,
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            limit_price=50000.0,
            exchange_id="binance"
        )
        result = self.validator._check_exchange_specific_limits(order)
        self.assertEqual(result.level, RiskCheckLevel.INFO)

        # Order exceeding exchange limits
        order.quantity = 250
        result = self.validator._check_exchange_specific_limits(order)
        self.assertEqual(result.level, RiskCheckLevel.ERROR)
        self.assertIn("exceeds exchange limit", result.message)

        # Exchange that is not allowed
        order.exchange_id = "kraken"
        result = self.validator._check_exchange_specific_limits(order)
        self.assertEqual(result.level, RiskCheckLevel.ERROR)
        self.assertIn("not allowed", result.message)

    def test_check_strategy_specific_limits(self):
        """Test the strategy-specific limits validation."""
        # Regular strategy
        result = self.validator._check_strategy_specific_limits(self.base_order)
        self.assertEqual(result.level, RiskCheckLevel.INFO)

        # Order exceeding strategy position size
        order = Order(
            order_id="test-order-018",
            symbol="SPY",
            quantity=4000,
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            limit_price=400.0,
            params={"strategy": "momentum", "equity": 100000}
        )
        result = self.validator._check_strategy_specific_limits(order)
        self.assertEqual(result.level, RiskCheckLevel.ERROR)
        self.assertIn("exceeds strategy limit", result.message)

        # Order exceeding strategy equity percentage
        order.quantity = 300
        order.limit_price = 5000.0  # 300 * 5000 = 1.5M, which is > 10% of 100k equity
        result = self.validator._check_strategy_specific_limits(order)
        self.assertEqual(result.level, RiskCheckLevel.ERROR)
        self.assertIn("exceeds strategy limit", result.message)

        # Strategy that is disabled
        self.validator.strategy_limits["momentum"]["enabled"] = False
        result = self.validator._check_strategy_specific_limits(self.base_order)
        self.assertEqual(result.level, RiskCheckLevel.ERROR)
        self.assertIn("disabled", result.message)

    def test_check_total_exposure(self):
        """Test the total exposure validation."""
        # Mock position manager and account manager
        position_manager = Mock()
        position_manager.get_total_position_value.return_value = 200000
        position_manager.get_market_price.return_value = 400.0

        account_manager = Mock()
        account_manager.get_equity.return_value = 100000

        # Order within exposure limits
        result = self.validator._check_total_exposure(self.base_order, position_manager, account_manager)
        self.assertEqual(result.level, RiskCheckLevel.INFO)

        # Order exceeding exposure limits
        position_manager.get_total_position_value.return_value = 280000
        result = self.validator._check_total_exposure(self.base_order, position_manager, account_manager)
        self.assertEqual(result.level, RiskCheckLevel.ERROR)
        self.assertIn("exceeds maximum", result.message)

    def test_check_open_orders_limit(self):
        """Test the open orders limit validation."""
        # Mock position manager
        position_manager = Mock()
        mock_position = {"SPY": Mock(open_orders=[])}
        position_manager.get_all_positions.return_value = mock_position

        # No open orders
        result = self.validator._check_open_orders_limit(self.base_order, position_manager)
        self.assertEqual(result.level, RiskCheckLevel.INFO)

        # Open orders within limit
        mock_position["SPY"].open_orders = [Mock() for _ in range(5)]
        result = self.validator._check_open_orders_limit(self.base_order, position_manager)
        self.assertEqual(result.level, RiskCheckLevel.INFO)

        # Open orders exceeding limit
        mock_position["SPY"].open_orders = [Mock() for _ in range(12)]
        result = self.validator._check_open_orders_limit(self.base_order, position_manager)
        self.assertEqual(result.level, RiskCheckLevel.ERROR)
        self.assertIn("exceeds maximum", result.message)

    def test_check_available_capital(self):
        """Test the available capital validation."""
        # Mock account manager
        account_manager = Mock()
        account_manager.get_available_capital.return_value = 50000
        account_manager.get_market_price.return_value = 400.0

        # Order within available capital
        result = self.validator._check_available_capital(self.base_order, account_manager)
        self.assertEqual(result.level, RiskCheckLevel.INFO)

        # Order exceeding available capital
        order = Order(
            order_id="test-order-019",
            symbol="SPY",
            quantity=200,
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            limit_price=300.0
        )
        account_manager.get_available_capital.return_value = 10000
        result = self.validator._check_available_capital(order, account_manager)
        self.assertEqual(result.level, RiskCheckLevel.ERROR)
        self.assertIn("exceeds available capital", result.message)

    def test_check_leverage_limits(self):
        """Test the leverage limits validation."""
        # Mock account manager
        account_manager = Mock()
        account_manager.get_current_leverage.return_value = 2.0

        # Leverage within limits
        result = self.validator._check_leverage_limits(self.base_order, account_manager)
        self.assertEqual(result.level, RiskCheckLevel.INFO)

        # Leverage exceeding limits
        account_manager.get_current_leverage.return_value = 7.0
        result = self.validator._check_leverage_limits(self.base_order, account_manager)
        self.assertEqual(result.level, RiskCheckLevel.ERROR)
        self.assertIn("exceeds maximum", result.message)

    def test_check_margin_requirements(self):
        """Test the margin requirements validation."""
        # Mock account manager
        account_manager = Mock()
        account_manager.check_margin_requirement = Mock(return_value={"passed": True})

        # Margin requirements satisfied
        result = self.validator._check_margin_requirements(self.base_order, account_manager)
        self.assertEqual(result.level, RiskCheckLevel.INFO)

        # Margin requirements not satisfied
        account_manager.check_margin_requirement.return_value = {"passed": False, "reason": "Insufficient margin"}
        result = self.validator._check_margin_requirements(self.base_order, account_manager)
        self.assertEqual(result.level, RiskCheckLevel.ERROR)
        self.assertIn("Insufficient margin", result.message)

    def test_check_market_conditions(self):
        """Test the market conditions validation."""
        # Mock market data service
        market_data_service = Mock()
        market_data_service.get_volatility.return_value = 0.02
        market_data_service.get_liquidity.return_value = 5000000

        # Normal market conditions
        result = self.validator._check_market_conditions(self.base_order, market_data_service)
        self.assertEqual(result.level, RiskCheckLevel.INFO)

        # High volatility
        market_data_service.get_volatility.return_value = 0.08
        result = self.validator._check_market_conditions(self.base_order, market_data_service)
        self.assertEqual(result.level, RiskCheckLevel.WARNING)
        self.assertIn("Extreme volatility", result.message)

        # Low liquidity
        market_data_service.get_volatility.return_value = 0.02
        market_data_service.get_liquidity.return_value = 500000
        result = self.validator._check_market_conditions(self.base_order, market_data_service)
        self.assertEqual(result.level, RiskCheckLevel.WARNING)
        self.assertIn("Low liquidity", result.message)

        # Check with regime info
        regime_info = {"volatility_regime": "high"}
        result = self.validator._check_market_conditions(self.base_order, market_data_service, regime_info)
        self.assertEqual(result.level, RiskCheckLevel.WARNING)
        self.assertIn("High volatility regime", result.message)

    def test_check_black_swan_limits(self):
        """Test the black swan limits validation."""
        # No black swan
        regime_info = {"black_swan_detected": False}
        result = self.validator._check_black_swan_limits(self.base_order, regime_info)
        self.assertEqual(result.level, RiskCheckLevel.INFO)

        # Market crash with long position
        regime_info = {"black_swan_detected": True, "black_swan_type": "market_crash"}
        result = self.validator._check_black_swan_limits(self.base_order, regime_info)
        self.assertEqual(result.level, RiskCheckLevel.ERROR)
        self.assertIn("restricted during market crash", result.message)

        # Market crash with short position
        self.base_order.side = OrderSide.SELL
        result = self.validator._check_black_swan_limits(self.base_order, regime_info)
        self.assertEqual(result.level, RiskCheckLevel.WARNING)
        self.assertIn("Trading during black swan event", result.message)

        # Flash crash
        regime_info = {"black_swan_detected": True, "black_swan_type": "flash_crash"}
        result = self.validator._check_black_swan_limits(self.base_order, regime_info)
        self.assertEqual(result.level, RiskCheckLevel.ERROR)
        self.assertIn("All trading restricted", result.message)

        # Liquidity crisis with small order
        regime_info = {"black_swan_detected": True, "black_swan_type": "liquidity_crisis"}
        result = self.validator._check_black_swan_limits(self.base_order, regime_info)
        self.assertEqual(result.level, RiskCheckLevel.WARNING)
        self.assertIn("Trading during black swan event", result.message)

        # Liquidity crisis with large order
        self.base_order.quantity = 300
        result = self.validator._check_black_swan_limits(self.base_order, regime_info)
        self.assertEqual(result.level, RiskCheckLevel.ERROR)
        self.assertIn("exceeds black swan limit", result.message)

    def test_apply_regime_adjustments(self):
        """Test applying regime-specific adjustments to risk limits."""
        # No regime info
        adjusted_limits = self.validator._apply_regime_adjustments(None)
        self.assertEqual(adjusted_limits["max_order_size"], 1000)
        self.assertEqual(adjusted_limits["price_deviation_threshold"], 0.05)

        # High volatility regime
        regime_info = {"volatility_regime": "high"}
        adjusted_limits = self.validator._apply_regime_adjustments(regime_info)
        self.assertEqual(adjusted_limits["max_order_size"], 500)  # 1000 * 0.5
        self.assertEqual(adjusted_limits["price_deviation_threshold"], 0.1)  # 0.05 * 2.0

        # Low volatility regime
        regime_info = {"volatility_regime": "low"}
        adjusted_limits = self.validator._apply_regime_adjustments(regime_info)
        self.assertEqual(adjusted_limits["max_order_size"], 1200)  # 1000 * 1.2
        self.assertEqual(adjusted_limits["price_deviation_threshold"], 0.04)  # 0.05 * 0.8

        # Black swan detection
        regime_info = {"volatility_regime": "medium", "black_swan_detected": True}
        adjusted_limits = self.validator._apply_regime_adjustments(regime_info)
        self.assertEqual(adjusted_limits["max_order_size"], 500)  # 1000 * 0.5
        self.assertEqual(adjusted_limits["price_deviation_threshold"], 0.1)  # 0.05 * 2.0

    def test_record_order_for_rate_limiting(self):
        """Test recording orders for rate limiting."""
        # Initial state
        self.assertEqual(len(self.validator._recent_orders.get("SPY", [])), 0)

        # Record an order
        self.validator._record_order_for_rate_limiting(self.base_order)
        self.assertEqual(len(self.validator._recent_orders.get("SPY", [])), 1)

        # Record more orders
        for _ in range(5):
            self.validator._record_order_for_rate_limiting(self.base_order)
        self.assertEqual(len(self.validator._recent_orders.get("SPY", [])), 6)

    def test_register_custom_check(self):
        """Test registering custom risk check."""
        # Define a custom check
        def custom_risk_check(order, *args):
            if order.quantity > 500:
                return RiskCheckResult(
                    level=RiskCheckLevel.ERROR,
                    message="Custom check failed",
                    check_name="custom_check"
                )
            return RiskCheckResult(
                level=RiskCheckLevel.INFO,
                message="Custom check passed",
                check_name="custom_check"
            )

        # Register the custom check
        self.validator.register_custom_check(custom_risk_check)
        self.assertEqual(len(self.validator._custom_checks), 1)

        # Create orders
        order_pass = Order(
            order_id="test-order-pass",
            symbol="SPY",
            quantity=100,
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            limit_price=400.0
        )

        order_fail = Order(
            order_id="test-order-fail",
            symbol="SPY",
            quantity=600,
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            limit_price=400.0
        )

        # Mock dependencies for validate_order
        position_manager = Mock()
        market_data_service = Mock()
        market_data_service.get_last_price.return_value = 400.0
        market_data_service.is_market_open.return_value = True
        market_data_service.get_volatility.return_value = 0.02
        market_data_service.get_liquidity.return_value = 5000000

        # Test validate_order with custom check
        results = self.validator.validate_order(order_pass, position_manager, market_data_service)
        custom_results = [r for r in results if r.check_name == "custom_check"]
        self.assertEqual(len(custom_results), 1)
        self.assertEqual(custom_results[0].level, RiskCheckLevel.INFO)

        results = self.validator.validate_order(order_fail, position_manager, market_data_service)
        custom_results = [r for r in results if r.check_name == "custom_check"]
        self.assertEqual(len(custom_results), 1)
        self.assertEqual(custom_results[0].level, RiskCheckLevel.ERROR)

    def test_dynamic_limit_updates(self):
        """Test dynamically updating risk limits."""
        # Initial limits
        self.assertEqual(self.validator.risk_limits["max_order_size"], 1000)

        # Update global limits
        new_limits = {"max_order_size": 1500, "max_leverage": 10}
        self.validator.update_limits(new_limits)
        self.assertEqual(self.validator.risk_limits["max_order_size"], 1500)
        self.assertEqual(self.validator.risk_limits["max_leverage"], 10)

        # Update instrument limits
        new_instrument_limits = {"max_order_size": 20, "max_position_size": 100}
        self.validator.update_instrument_limits("BTC-USD", new_instrument_limits)
        self.assertEqual(self.validator.instrument_limits["BTC-USD"]["max_order_size"], 20)
        self.assertEqual(self.validator.instrument_limits["BTC-USD"]["max_position_size"], 100)

        # Update exchange limits
        new_exchange_limits = {"max_order_size": 300, "allowed": True}
        self.validator.update_exchange_limits("binance", new_exchange_limits)
        self.assertEqual(self.validator.exchange_limits["binance"]["max_order_size"], 300)
        self.assertEqual(self.validator.exchange_limits["binance"]["allowed"], True)

        # Update strategy limits
        new_strategy_limits = {"max_position_size": 4000, "max_equity_percent": 0.15}
        self.validator.update_strategy_limits("momentum", new_strategy_limits)
        self.assertEqual(self.validator.strategy_limits["momentum"]["max_position_size"], 4000)
        self.assertEqual(self.validator.strategy_limits["momentum"]["max_equity_percent"], 0.15)

        # Update regime adjustments
        new_regime_adjustments = {
            "high_volatility": {
                "max_order_size_multiplier": 0.3,
                "price_deviation_threshold_multiplier": 3.0
            }
        }
        self.validator.update_regime_adjustments(new_regime_adjustments)
        self.assertEqual(
            self.validator._regime_adjustments["high_volatility"]["max_order_size_multiplier"],
            0.3
        )
        self.assertEqual(
            self.validator._regime_adjustments["high_volatility"]["price_deviation_threshold_multiplier"],
            3.0
        )

    def test_reset(self):
        """Test resetting validator state."""
        # Modify some settings
        self.validator.risk_limits["max_order_size"] = 1500
        self.validator.update_instrument_limits("NEW-COIN", {"max_order_size": 50})

        # Add some recent orders
        for _ in range(5):
            self.validator._record_order_for_rate_limiting(self.base_order)

        # Reset validator
        self.validator.reset()

        # Check if defaults are restored
        self.assertEqual(self.validator.risk_limits["max_order_size"], 1000)
        self.assertFalse("NEW-COIN" in self.validator.instrument_limits)
        self.assertEqual(len(self.validator._recent_orders.get("SPY", [])), 0)


if __name__ == "__main__":
    unittest.main()