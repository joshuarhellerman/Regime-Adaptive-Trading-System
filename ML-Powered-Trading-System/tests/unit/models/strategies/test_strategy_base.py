import unittest
from unittest import mock
import pandas as pd
import numpy as np
import uuid
import time
from datetime import datetime

from models.strategies.strategy_base import TradingStrategy, DirectionalBias, VolatilityRegime


class TestTradingStrategy(unittest.TestCase):
    """Test cases for the TradingStrategy base class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.strategy_name = "test_strategy"
        self.parameters = {"param1": 10, "param2": "value"}
        self.strategy = TradingStrategy(self.strategy_name, self.parameters)

        # Create sample market data for testing
        self.data = pd.DataFrame({
            'open': [1.1, 1.2, 1.3, 1.4, 1.5],
            'high': [1.15, 1.25, 1.35, 1.45, 1.55],
            'low': [1.05, 1.15, 1.25, 1.35, 1.45],
            'close': [1.12, 1.22, 1.32, 1.42, 1.52]
        }, index=pd.date_range(start="2023-01-01", periods=5, freq="D"))

    def test_initialization(self):
        """Test the initialization of the TradingStrategy class."""
        self.assertEqual(self.strategy.name, self.strategy_name)
        self.assertEqual(self.strategy.parameters, self.parameters)
        self.assertTrue(self.strategy.id.startswith(f"{self.strategy_name.lower()}-"))
        self.assertEqual(len(self.strategy.id.split("-")[1]), 8)  # UUID part should be 8 chars

        # Test with custom strategy_id
        custom_id = "custom-id-123"
        strategy = TradingStrategy(self.strategy_name, self.parameters, strategy_id=custom_id)
        self.assertEqual(strategy.id, custom_id)

    def test_initialization_default_values(self):
        """Test that default values are set correctly during initialization."""
        self.assertEqual(self.strategy.stop_loss_modifier, 1.0)
        self.assertEqual(self.strategy.profit_target_modifier, 1.0)
        self.assertEqual(self.strategy.position_size_multiplier, 1.0)
        self.assertEqual(self.strategy.max_position_pct, 0.05)
        self.assertEqual(self.strategy.account_balance, 100000)
        self.assertEqual(self.strategy.trend_filter_strength, 1.0)
        self.assertEqual(self.strategy.signal_threshold_modifier, 1.0)
        self.assertIsNone(self.strategy.regime_characteristics)
        self.assertIsNone(self.strategy.currency_pair)
        self.assertEqual(self.strategy.last_signal_time, 0)
        self.assertEqual(self.strategy.min_time_between_signals, 3600)

    def test_validate_data_success(self):
        """Test successful data validation."""
        self.strategy.validate_data(self.data)  # Should not raise an exception

    def test_validate_data_missing_columns(self):
        """Test data validation with missing columns."""
        incomplete_data = pd.DataFrame({
            'open': [1.1, 1.2],
            'high': [1.15, 1.25],
            # Missing 'low' and 'close' columns
        })

        with self.assertRaises(ValueError) as context:
            self.strategy.validate_data(incomplete_data)

        self.assertTrue("missing required columns" in str(context.exception).lower())
        self.assertTrue("low" in str(context.exception))
        self.assertTrue("close" in str(context.exception))

    def test_generate_signal_base_implementation(self):
        """Test the base implementation of generate_signal."""
        current_time = time.time()
        result = self.strategy.generate_signal(self.data)

        self.assertIsNone(result)  # Base implementation returns None
        self.assertGreaterEqual(self.strategy.last_signal_time, current_time)  # Timestamp updated

    def test_risk_parameters_base_implementation(self):
        """Test the base implementation of risk_parameters."""
        entry_price = 1.5
        result = self.strategy.risk_parameters(self.data, entry_price)

        expected = {
            'stop_loss_pct': 0.02 * self.strategy.stop_loss_modifier,
            'take_profit_pct': 0.04 * self.strategy.profit_target_modifier,
            'position_size': self.strategy.dynamic_position_size(self.strategy.account_balance)
        }

        self.assertEqual(result, expected)

    def test_exit_signal_base_implementation(self):
        """Test the base implementation of exit_signal."""
        position = {'entry_price': 1.5, 'type': 'long'}
        result = self.strategy.exit_signal(self.data, position)

        self.assertFalse(result)  # Base implementation returns False

    def test_dynamic_position_size(self):
        """Test dynamic position size calculation."""
        equity = 200000
        expected = equity * self.strategy.max_position_pct * self.strategy.position_size_multiplier
        result = self.strategy.dynamic_position_size(equity)

        self.assertEqual(result, expected)

        # Test with modified position size multiplier
        self.strategy.position_size_multiplier = 0.5
        expected *= 0.5
        result = self.strategy.dynamic_position_size(equity)

        self.assertEqual(result, expected)

    def test_get_required_features(self):
        """Test get_required_features returns the base set of features."""
        expected = {'open', 'high', 'low', 'close'}
        result = self.strategy.get_required_features()

        self.assertEqual(result, expected)

    def test_adapt_to_regime(self):
        """Test adapt_to_regime updates regime characteristics."""
        regime_data = {
            'currency_pair': 'EUR/USD',
            'volatility': 'high',
            'direction': 'upward'
        }

        # Mock the optimization methods
        with mock.patch.object(self.strategy, '_optimize_for_time_of_day') as mock_time, \
                mock.patch.object(self.strategy, '_optimize_for_directional_bias') as mock_bias, \
                mock.patch.object(self.strategy, '_optimize_for_volatility_regime') as mock_vol, \
                mock.patch.object(self.strategy, '_optimize_for_currency_pair') as mock_pair:
            self.strategy.adapt_to_regime(regime_data)

            # Verify regime characteristics are stored
            self.assertEqual(self.strategy.regime_characteristics, regime_data)
            self.assertEqual(self.strategy.currency_pair, 'EUR/USD')

            # Verify optimization methods were called
            mock_time.assert_called_once()
            mock_bias.assert_called_once()
            mock_vol.assert_called_once()
            mock_pair.assert_called_once()

    def test_log_signal(self):
        """Test log_signal functionality."""
        with mock.patch('models.strategies.strategy_base.logger') as mock_logger:
            signal_type = 'long'
            reason = 'test reason'
            self.strategy.log_signal(signal_type, self.data, reason)

            # Verify logger was called with appropriate message
            mock_logger.info.assert_called_once()
            log_message = mock_logger.info.call_args[0][0]

            self.assertTrue(self.strategy.name in log_message)
            self.assertTrue(signal_type.upper() in log_message)
            self.assertTrue(str(self.data['close'].iloc[-1]) in log_message)
            self.assertTrue(reason in log_message)

    def test_cluster_fit_base_implementation(self):
        """Test the base implementation of cluster_fit."""
        cluster_metrics = {'volatility': 0.01, 'trend': 0.5}
        result = self.strategy.cluster_fit(cluster_metrics)

        self.assertEqual(result, 0.5)  # Base implementation returns 0.5

    def test_on_trade_completed_base_implementation(self):
        """Test the base implementation of on_trade_completed."""
        trade_result = {'profit': 100, 'pips': 10}
        # Should not raise exception
        self.strategy.on_trade_completed(trade_result)

    def test_update_parameters_online_base_implementation(self):
        """Test the base implementation of update_parameters_online."""
        performance_metrics = {'win_rate': 0.6, 'profit_factor': 1.5}
        market_conditions = {'volatility': 'high'}
        # Should not raise exception
        self.strategy.update_parameters_online(performance_metrics, market_conditions)

    def test_optimization_methods_base_implementations(self):
        """Test the base implementations of optimization methods."""
        # These methods don't return anything in the base class, but should run without errors
        self.strategy._optimize_for_time_of_day()
        self.strategy._optimize_for_directional_bias()
        self.strategy._optimize_for_volatility_regime()
        self.strategy._optimize_for_currency_pair()

    def test_directional_bias_enum(self):
        """Test the DirectionalBias enum."""
        self.assertTrue(hasattr(DirectionalBias, 'NEUTRAL'))
        self.assertTrue(hasattr(DirectionalBias, 'UPWARD'))
        self.assertTrue(hasattr(DirectionalBias, 'DOWNWARD'))

    def test_volatility_regime_enum(self):
        """Test the VolatilityRegime enum."""
        self.assertTrue(hasattr(VolatilityRegime, 'LOW'))
        self.assertTrue(hasattr(VolatilityRegime, 'NORMAL'))
        self.assertTrue(hasattr(VolatilityRegime, 'HIGH'))


if __name__ == '__main__':
    unittest.main()