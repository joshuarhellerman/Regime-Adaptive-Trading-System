import unittest
from unittest import mock
import pandas as pd
import numpy as np
from datetime import datetime

from models.strategies.strategy_context import StrategyContext
from models.strategies.strategy_base import TradingStrategy, DirectionalBias, VolatilityRegime


class TestStrategyContext(unittest.TestCase):
    """Test cases for the StrategyContext class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a mock strategy
        self.mock_strategy = mock.Mock(spec=TradingStrategy)
        self.mock_strategy.name = "mock_strategy"
        self.mock_strategy.id = "mock_strategy-12345678"
        self.mock_strategy.parameters = {"param1": 10, "param2": "value"}

        # Create a strategy context with the mock strategy
        self.context = StrategyContext(self.mock_strategy)

        # Create sample market data for testing
        self.data = pd.DataFrame({
            'open': [1.1, 1.2, 1.3, 1.4, 1.5],
            'high': [1.15, 1.25, 1.35, 1.45, 1.55],
            'low': [1.05, 1.15, 1.25, 1.35, 1.45],
            'close': [1.12, 1.22, 1.32, 1.42, 1.52]
        }, index=pd.date_range(start="2023-01-01", periods=5, freq="D"))

    def test_initialization(self):
        """Test the initialization of the StrategyContext class."""
        self.assertEqual(self.context.strategy, self.mock_strategy)
        self.assertIsNone(self.context.current_position)
        self.assertEqual(self.context.trade_history, [])
        self.assertEqual(self.context.performance_metrics, {})
        self.assertEqual(self.context.risk_parameters, {})

    def test_set_strategy(self):
        """Test setting a new strategy."""
        new_strategy = mock.Mock(spec=TradingStrategy)
        new_strategy.name = "new_strategy"
        new_strategy.id = "new_strategy-87654321"

        self.context.set_strategy(new_strategy)
        self.assertEqual(self.context.strategy, new_strategy)

    def test_process_market_data_no_position(self):
        """Test processing market data when no position is open."""
        # Configure mock strategy to generate a signal
        self.mock_strategy.generate_signal.return_value = {
            'type': 'long',
            'reason': 'test signal',
            'strength': 0.8
        }
        self.mock_strategy.risk_parameters.return_value = {
            'stop_loss_pct': 0.02,
            'take_profit_pct': 0.04,
            'position_size': 5000
        }

        result = self.context.process_market_data(self.data)

        # Verify strategy methods were called
        self.mock_strategy.generate_signal.assert_called_once_with(self.data)
        self.mock_strategy.risk_parameters.assert_called_once()

        entry_price = self.data['close'].iloc[-1]
        expected_position = {
            'entry_price': entry_price,
            'type': 'long',
            'size': 5000,
            'stop_loss': entry_price * (1 - 0.02),
            'take_profit': entry_price * (1 + 0.04),
            'entry_time': mock.ANY,
            'reason': 'test signal',
            'strength': 0.8
        }

        # Check that the position was opened and result contains expected data
        self.assertIsNotNone(self.context.current_position)
        for key in expected_position:
            if key != 'entry_time':  # Skip time comparison
                self.assertEqual(self.context.current_position[key], expected_position[key])

        self.assertEqual(result['action'], 'OPEN_POSITION')
        self.assertEqual(result['position_type'], 'long')
        self.assertEqual(result['price'], entry_price)

    def test_process_market_data_with_position(self):
        """Test processing market data when a position is already open."""
        # Set up an existing position
        entry_price = 1.42
        self.context.current_position = {
            'entry_price': entry_price,
            'type': 'long',
            'size': 5000,
            'stop_loss': entry_price * 0.98,
            'take_profit': entry_price * 1.04,
            'entry_time': datetime.now(),
            'reason': 'test signal',
            'strength': 0.8
        }

        # Configure mock strategy to not exit
        self.mock_strategy.exit_signal.return_value = False

        # Process data
        result = self.context.process_market_data(self.data)

        # Verify exit_signal was called
        self.mock_strategy.exit_signal.assert_called_once_with(self.data, self.context.current_position)

        # Verify position is still open and no new signal was generated
        self.assertIsNotNone(self.context.current_position)
        self.mock_strategy.generate_signal.assert_not_called()
        self.assertEqual(result['action'], 'HOLD_POSITION')

    def test_process_market_data_with_exit_signal(self):
        """Test processing market data when exit signal is generated."""
        # Set up an existing position
        entry_price = 1.42
        position_type = 'long'
        self.context.current_position = {
            'entry_price': entry_price,
            'type': position_type,
            'size': 5000,
            'stop_loss': entry_price * 0.98,
            'take_profit': entry_price * 1.04,
            'entry_time': datetime.now(),
            'reason': 'test signal',
            'strength': 0.8
        }

        # Configure mock strategy to exit
        self.mock_strategy.exit_signal.return_value = True

        # Process data
        result = self.context.process_market_data(self.data)

        # Verify exit_signal was called
        self.mock_strategy.exit_signal.assert_called_once_with(self.data, self.context.current_position)

        # Verify position was closed
        self.assertIsNone(self.context.current_position)
        self.mock_strategy.generate_signal.assert_not_called()
        self.assertEqual(result['action'], 'CLOSE_POSITION')
        self.assertEqual(result['position_type'], position_type)
        self.assertEqual(result['price'], self.data['close'].iloc[-1])

        # Verify trade was added to history
        self.assertEqual(len(self.context.trade_history), 1)
        trade = self.context.trade_history[0]
        self.assertEqual(trade['entry_price'], entry_price)
        self.assertEqual(trade['exit_price'], self.data['close'].iloc[-1])
        self.assertEqual(trade['type'], position_type)

    def test_process_market_data_stop_loss_hit(self):
        """Test processing market data when stop loss is hit."""
        # Set up an existing position
        entry_price = 1.42
        stop_loss = 1.40
        self.context.current_position = {
            'entry_price': entry_price,
            'type': 'long',
            'size': 5000,
            'stop_loss': stop_loss,
            'take_profit': entry_price * 1.04,
            'entry_time': datetime.now(),
            'reason': 'test signal',
            'strength': 0.8
        }

        # Create data with the low below stop loss
        data_with_stop_loss_hit = pd.DataFrame({
            'open': [1.42],
            'high': [1.43],
            'low': [1.39],  # Below stop loss
            'close': [1.41]
        }, index=pd.date_range(start="2023-01-06", periods=1, freq="D"))

        # Process data
        result = self.context.process_market_data(data_with_stop_loss_hit)

        # Verify position was closed
        self.assertIsNone(self.context.current_position)
        self.assertEqual(result['action'], 'STOP_LOSS')
        self.assertEqual(result['position_type'], 'long')
        self.assertEqual(result['price'], stop_loss)

        # Verify trade was added to history
        self.assertEqual(len(self.context.trade_history), 1)
        trade = self.context.trade_history[0]
        self.assertEqual(trade['exit_reason'], 'stop_loss')

    def test_process_market_data_take_profit_hit(self):
        """Test processing market data when take profit is hit."""
        # Set up an existing position
        entry_price = 1.42
        take_profit = 1.48
        self.context.current_position = {
            'entry_price': entry_price,
            'type': 'long',
            'size': 5000,
            'stop_loss': entry_price * 0.98,
            'take_profit': take_profit,
            'entry_time': datetime.now(),
            'reason': 'test signal',
            'strength': 0.8
        }

        # Create data with the high above take profit
        data_with_take_profit_hit = pd.DataFrame({
            'open': [1.45],
            'high': [1.49],  # Above take profit
            'low': [1.44],
            'close': [1.46]
        }, index=pd.date_range(start="2023-01-06", periods=1, freq="D"))

        # Process data
        result = self.context.process_market_data(data_with_take_profit_hit)

        # Verify position was closed
        self.assertIsNone(self.context.current_position)
        self.assertEqual(result['action'], 'TAKE_PROFIT')
        self.assertEqual(result['position_type'], 'long')
        self.assertEqual(result['price'], take_profit)

        # Verify trade was added to history
        self.assertEqual(len(self.context.trade_history), 1)
        trade = self.context.trade_history[0]
        self.assertEqual(trade['exit_reason'], 'take_profit')

    def test_process_market_data_short_position(self):
        """Test processing market data for a short position."""
        # Configure mock strategy to generate a short signal
        self.mock_strategy.generate_signal.return_value = {
            'type': 'short',
            'reason': 'test short signal',
            'strength': 0.7
        }
        self.mock_strategy.risk_parameters.return_value = {
            'stop_loss_pct': 0.02,
            'take_profit_pct': 0.04,
            'position_size': 4000
        }

        result = self.context.process_market_data(self.data)

        entry_price = self.data['close'].iloc[-1]
        expected_position = {
            'entry_price': entry_price,
            'type': 'short',
            'size': 4000,
            'stop_loss': entry_price * (1 + 0.02),  # For short position, stop loss is above entry
            'take_profit': entry_price * (1 - 0.04),  # For short position, take profit is below entry
            'entry_time': mock.ANY,
            'reason': 'test short signal',
            'strength': 0.7
        }

        # Check that the position was opened and result contains expected data
        self.assertIsNotNone(self.context.current_position)
        for key in expected_position:
            if key != 'entry_time':
                self.assertEqual(self.context.current_position[key], expected_position[key])

        self.assertEqual(result['action'], 'OPEN_POSITION')
        self.assertEqual(result['position_type'], 'short')

    def test_calculate_performance_metrics(self):
        """Test calculation of performance metrics."""
        # Add some trades to the history
        self.context.trade_history = [
            {
                'entry_price': 1.40,
                'exit_price': 1.45,
                'type': 'long',
                'size': 5000,
                'entry_time': datetime(2023, 1, 1, 10, 0),
                'exit_time': datetime(2023, 1, 2, 10, 0),
                'exit_reason': 'take_profit',
                'pips': 50,
                'profit': 250
            },
            {
                'entry_price': 1.42,
                'exit_price': 1.40,
                'type': 'long',
                'size': 5000,
                'entry_time': datetime(2023, 1, 3, 10, 0),
                'exit_time': datetime(2023, 1, 4, 10, 0),
                'exit_reason': 'stop_loss',
                'pips': -20,
                'profit': -100
            },
            {
                'entry_price': 1.38,
                'exit_price': 1.35,
                'type': 'short',
                'size': 4000,
                'entry_time': datetime(2023, 1, 5, 10, 0),
                'exit_time': datetime(2023, 1, 6, 10, 0),
                'exit_reason': 'take_profit',
                'pips': 30,
                'profit': 120
            }
        ]

        self.context.calculate_performance_metrics()

        metrics = self.context.performance_metrics
        self.assertEqual(metrics['total_trades'], 3)
        self.assertEqual(metrics['winning_trades'], 2)
        self.assertEqual(metrics['losing_trades'], 1)
        self.assertAlmostEqual(metrics['win_rate'], 2 / 3)
        self.assertEqual(metrics['total_profit'], 270)
        self.assertEqual(metrics['total_pips'], 60)
        self.assertEqual(metrics['average_profit_per_trade'], 90)
        self.assertEqual(metrics['average_pips_per_trade'], 20)
        self.assertAlmostEqual(metrics['profit_factor'], (250 + 120) / 100)

        # Verify strategy was updated with performance metrics
        self.mock_strategy.update_parameters_online.assert_called_once()

    def test_close_position(self):
        """Test closing a position manually."""
        # Set up an existing position
        entry_price = 1.42
        self.context.current_position = {
            'entry_price': entry_price,
            'type': 'long',
            'size': 5000,
            'stop_loss': entry_price * 0.98,
            'take_profit': entry_price * 1.04,
            'entry_time': datetime.now(),
            'reason': 'test signal',
            'strength': 0.8
        }

        exit_price = 1.44
        result = self.context.close_position(exit_price, 'manual')

        # Verify position was closed
        self.assertIsNone(self.context.current_position)
        self.assertEqual(result['action'], 'CLOSE_POSITION')
        self.assertEqual(result['position_type'], 'long')
        self.assertEqual(result['price'], exit_price)

        # Verify trade was added to history
        self.assertEqual(len(self.context.trade_history), 1)
        trade = self.context.trade_history[0]
        self.assertEqual(trade['exit_reason'], 'manual')
        self.assertEqual(trade['exit_price'], exit_price)

        # Calculate expected profit and pips
        pips = (exit_price - entry_price) * 10000  # Assuming 4 decimal forex
        profit = pips * 5000 / 10000  # size * pip value

        self.assertEqual(trade['pips'], pips)
        self.assertEqual(trade['profit'], profit)

    def test_update_risk_parameters(self):
        """Test updating risk parameters."""
        new_params = {
            'stop_loss_pct': 0.03,
            'take_profit_pct': 0.06,
            'position_size': 6000
        }

        self.context.update_risk_parameters(new_params)
        self.assertEqual(self.context.risk_parameters, new_params)

    def test_adapt_to_market_conditions(self):
        """Test adapting strategy to market conditions."""
        market_conditions = {
            'currency_pair': 'EUR/USD',
            'volatility': 'high',
            'direction': 'upward',
            'time_of_day': '12:00'
        }

        self.context.adapt_to_market_conditions(market_conditions)

        # Verify strategy was adapted
        self.mock_strategy.adapt_to_regime.assert_called_once_with(market_conditions)


if __name__ == '__main__':
    unittest.main()