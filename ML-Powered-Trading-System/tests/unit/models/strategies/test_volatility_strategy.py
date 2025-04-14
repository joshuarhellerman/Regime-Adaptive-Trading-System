import unittest
from unittest.mock import patch, MagicMock, Mock
import pandas as pd
import numpy as np
from typing import Dict, Any
import time

# Import the class to test
from models.strategies.volatility_strategy import VolatilityStrategy
from models.strategies.strategy_base import DirectionalBias, VolatilityRegime


class TestVolatilityStrategy(unittest.TestCase):
    """Test suite for the VolatilityStrategy class."""

    def setUp(self):
        """Setup test fixtures before each test method."""
        # Create a strategy instance with default parameters
        self.strategy = VolatilityStrategy(name="TestVolatility", strategy_id="test-vol-1")
        
        # Mock EventBus to avoid actual event emissions
        self.event_bus_patcher = patch('models.strategies.volatility_strategy.EventBus')
        self.mock_event_bus = self.event_bus_patcher.start()
        
        # Create sample data for testing
        self.create_sample_data()
        
    def tearDown(self):
        """Clean up after each test method."""
        self.event_bus_patcher.stop()
        
    def create_sample_data(self):
        """Create sample OHLCV data for testing."""
        # Create a 100-bar DataFrame with OHLCV data
        dates = pd.date_range(start='2023-01-01', periods=100, freq='1h')
        
        # Create oscillating price data
        base_price = 100.0
        price_oscillation = np.sin(np.linspace(0, 6 * np.pi, 100)) * 5
        
        # Create DataFrame with OHLCV data
        self.data = pd.DataFrame({
            'open': base_price + price_oscillation + np.random.normal(0, 0.5, 100),
            'high': base_price + price_oscillation + 2 + np.random.normal(0, 0.8, 100),
            'low': base_price + price_oscillation - 2 + np.random.normal(0, 0.8, 100),
            'close': base_price + price_oscillation + np.random.normal(0, 0.5, 100),
            'volume': np.random.randint(1000, 10000, 100)
        }, index=dates)
        
        # Ensure high is always highest and low is always lowest
        self.data['high'] = self.data[['open', 'close', 'high']].max(axis=1)
        self.data['low'] = self.data[['open', 'close', 'low']].min(axis=1)
        
        # Create sample data with pre-calculated indicators
        self.data_with_indicators = self.data.copy()
        # Add Bollinger Bands
        bb_period = self.strategy.parameters['bb_period']
        bb_std = self.strategy.parameters['bb_std']
        
        self.data_with_indicators['bb_middle'] = self.data['close'].rolling(bb_period).mean()
        std_dev = self.data['close'].rolling(bb_period).std()
        self.data_with_indicators['bb_upper'] = self.data_with_indicators['bb_middle'] + (std_dev * bb_std)
        self.data_with_indicators['bb_lower'] = self.data_with_indicators['bb_middle'] - (std_dev * bb_std)
        
        # Add ATR
        atr_period = self.strategy.parameters['atr_period']
        tr1 = self.data['high'] - self.data['low']
        tr2 = abs(self.data['high'] - self.data['close'].shift())
        tr3 = abs(self.data['low'] - self.data['close'].shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        self.data_with_indicators['atr_14'] = tr.rolling(window=atr_period).mean()
        
        # Create data with volatility expansion
        self.expansion_data = self.data_with_indicators.copy()
        # Increase ATR for last 5 bars to simulate volatility expansion
        self.expansion_data.loc[self.expansion_data.index[-5:], 'atr_14'] *= 2.0
        
        # Create data with volatility contraction
        self.contraction_data = self.data_with_indicators.copy()
        # Decrease ATR for last 5 bars to simulate volatility contraction
        self.contraction_data.loc[self.contraction_data.index[-5:], 'atr_14'] *= 0.4
        
        # Sample position data for testing exit signals
        self.long_position = {
            'direction': 'long',
            'entry_price': 100.0,
            'entry_time': time.time() - 3600,  # 1 hour ago
            'size': 1.0,
            'id': 'pos-123',
            'entry_reason': 'breakout after volatility contraction'
        }
        
        self.short_position = {
            'direction': 'short',
            'entry_price': 100.0,
            'entry_time': time.time() - 3600,  # 1 hour ago
            'size': 1.0,
            'id': 'pos-456',
            'entry_reason': 'reversal after volatility expansion'
        }

    def test_initialization(self):
        """Test proper initialization of the strategy."""
        # Test that default parameters are properly set
        self.assertEqual(self.strategy.name, "TestVolatility")
        self.assertEqual(self.strategy.id, "test-vol-1")
        self.assertEqual(self.strategy.parameters['bb_period'], 20)
        self.assertEqual(self.strategy.parameters['atr_period'], 14)
        self.assertEqual(self.strategy.parameters['expansion_threshold'], 1.4)
        self.assertEqual(self.strategy.parameters['contraction_threshold'], 0.6)
        
        # Test state initialization
        self.assertEqual(self.strategy._state['volatility_state'], "normal")
        self.assertEqual(self.strategy._state['volatility_trend'], "neutral")
        self.assertEqual(self.strategy._state['volatility_threshold_modifier'], 1.0)
        
        # Test that event bus was subscribed
        self.mock_event_bus.subscribe.assert_any_call(f"strategy.{self.strategy.id}.parameter_update", 
                                                   self.strategy._handle_parameter_update)
        self.mock_event_bus.subscribe.assert_any_call("market.regime_change", 
                                                   self.strategy._handle_regime_change)

    def test_validate_parameters(self):
        """Test parameter validation logic."""
        # Test valid parameters (should not raise exception)
        self.strategy._validate_parameters()
        
        # Test invalid parameters
        with self.assertRaises(ValueError):
            # Test negative period
            self.strategy.parameters['bb_period'] = -5
            self.strategy._validate_parameters()
        
        # Reset for next test
        self.strategy.parameters['bb_period'] = 20
        
        with self.assertRaises(ValueError):
            # Test invalid contraction threshold
            self.strategy.parameters['contraction_threshold'] = 1.5  # Should be < 1.0
            self.strategy._validate_parameters()
        
        # Reset for next test
        self.strategy.parameters['contraction_threshold'] = 0.6
        
        with self.assertRaises(ValueError):
            # Test invalid volatility percentiles
            self.strategy.parameters['min_volatility_percentile'] = 0.8
            self.strategy.parameters['max_volatility_percentile'] = 0.5  # min should be < max
            self.strategy._validate_parameters()

    def test_calculate_bollinger_bands(self):
        """Test Bollinger Bands calculation."""
        # Calculate Bollinger Bands with test data
        upper, middle, lower = self.strategy._calculate_bollinger_bands(
            self.data['close'], window=20, num_std=2.0)
        
        # Check that the bands were calculated correctly
        self.assertEqual(len(upper), len(self.data))
        self.assertEqual(len(middle), len(self.data))
        self.assertEqual(len(lower), len(self.data))
        
        # Check that the middle band is the moving average
        pd.testing.assert_series_equal(middle, self.data['close'].rolling(window=20).mean())
        
        # Check that upper and lower bands are equidistant from middle
        std_dev = self.data['close'].rolling(window=20).std()
        pd.testing.assert_series_equal(upper, middle + (std_dev * 2.0), check_names=False)
        pd.testing.assert_series_equal(lower, middle - (std_dev * 2.0), check_names=False)

    def test_calculate_atr(self):
        """Test ATR calculation."""
        # Calculate ATR with test data
        atr = self.strategy._calculate_atr(self.data, period=14)
        
        # Check that ATR was calculated correctly
        self.assertEqual(len(atr), len(self.data))
        
        # Calculate manual ATR for comparison
        tr1 = self.data['high'] - self.data['low']
        tr2 = abs(self.data['high'] - self.data['close'].shift())
        tr3 = abs(self.data['low'] - self.data['close'].shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        expected_atr = tr.rolling(window=14).mean()
        
        # Compare with manual calculation
        pd.testing.assert_series_equal(atr, expected_atr)

    def test_update_volatility_state_normal(self):
        """Test volatility state detection with normal volatility."""
        # Use data with normal volatility
        self.strategy._update_volatility_state(self.data_with_indicators, self.data_with_indicators['atr_14'])
        
        # Check that state is set to normal
        self.assertEqual(self.strategy._state['volatility_state'], "normal")
        
        # Check event emission
        self.mock_event_bus.emit.assert_called_with(
            "volatility.state_update", 
            {
                'strategy_id': self.strategy.id,
                'timestamp': unittest.mock.ANY,
                'volatility_state': "normal",
                'volatility_trend': unittest.mock.ANY,
                'latest_atr': unittest.mock.ANY
            }
        )

    def test_update_volatility_state_expansion(self):
        """Test volatility state detection during expansion."""
        # Use data with volatility expansion
        self.strategy._update_volatility_state(self.expansion_data, self.expansion_data['atr_14'])
        
        # Check that state is set to expansion
        self.assertEqual(self.strategy._state['volatility_state'], "expansion")
        
        # Check event emission
        self.mock_event_bus.emit.assert_called_with(
            "volatility.state_update", 
            {
                'strategy_id': self.strategy.id,
                'timestamp': unittest.mock.ANY,
                'volatility_state': "expansion",
                'volatility_trend': unittest.mock.ANY,
                'latest_atr': unittest.mock.ANY
            }
        )

    def test_update_volatility_state_contraction(self):
        """Test volatility state detection during contraction."""
        # Use data with volatility contraction
        self.strategy._update_volatility_state(self.contraction_data, self.contraction_data['atr_14'])
        
        # Check that state is set to contraction
        self.assertEqual(self.strategy._state['volatility_state'], "contraction")
        
        # Check event emission
        self.mock_event_bus.emit.assert_called_with(
            "volatility.state_update", 
            {
                'strategy_id': self.strategy.id,
                'timestamp': unittest.mock.ANY,
                'volatility_state': "contraction",
                'volatility_trend': unittest.mock.ANY,
                'latest_atr': unittest.mock.ANY
            }
        )

    def test_generate_signal_insufficient_data(self):
        """Test signal generation with insufficient data."""
        # Create small dataset
        small_data = self.data_with_indicators.iloc[:10]
        
        # Generate signal with insufficient data
        signal = self.strategy.generate_signal(small_data)
        
        # Check that no signal is generated
        self.assertIsNone(signal)

    @patch('models.strategies.volatility_strategy.VolatilityStrategy.log_signal')
    def test_generate_signal_contraction_breakout(self, mock_log_signal):
        """Test signal generation during volatility contraction breakout."""
        # Set strategy state to contraction
        self.strategy._state['volatility_state'] = "contraction"
        self.strategy._state['volatility_threshold_modifier'] = 1.0
        
        # Modify data to create a breakout above upper band
        test_data = self.data_with_indicators.copy()
        test_data.loc[test_data.index[-1], 'close'] = test_data['bb_upper'].iloc[-1] * 1.05
        
        # Generate signal
        signal = self.strategy.generate_signal(test_data)
        
        # Check for long signal
        self.assertEqual(signal, "long")
        
        # Check that signal was logged
        mock_log_signal.assert_called_with('long', test_data, unittest.mock.ANY)
        
        # Check event emission
        self.mock_event_bus.emit.assert_any_call(
            "strategy.signal", 
            {
                'strategy_id': self.strategy.id,
                'strategy_type': 'volatility',
                'timestamp': unittest.mock.ANY,
                'signal': "long",
                'instrument': unittest.mock.ANY,
                'price': test_data['close'].iloc[-1],
                'confidence': unittest.mock.ANY,
                'metadata': unittest.mock.ANY
            }
        )

    @patch('models.strategies.volatility_strategy.VolatilityStrategy.log_signal')
    def test_generate_signal_expansion_reversal(self, mock_log_signal):
        """Test signal generation during volatility expansion reversal."""
        # Set strategy state to expansion
        self.strategy._state['volatility_state'] = "expansion"
        
        # Modify data to create a significant move below lower band
        test_data = self.data_with_indicators.copy()
        test_data.loc[test_data.index[-1], 'close'] = test_data['bb_lower'].iloc[-1] * 0.92
        
        # Generate signal
        signal = self.strategy.generate_signal(test_data)
        
        # Check for long signal (reversal from downward movement)
        self.assertEqual(signal, "long")
        
        # Check that signal was logged
        mock_log_signal.assert_called_with('long', test_data, unittest.mock.ANY)

    def test_exit_signal_momentum_stall(self):
        """Test exit signal generation on momentum stall."""
        # Set up data for momentum stall
        test_data = self.data_with_indicators.copy()
        test_data.loc[test_data.index[-3:], 'close'] = [105, 104, 103]  # Declining prices
        
        # Check exit signal for long position with breakout entry
        self.long_position['entry_reason'] = 'breakout after volatility contraction'
        
        # Generate exit signal
        exit_signal = self.strategy.exit_signal(test_data, self.long_position)
        
        # Check that exit signal is generated
        self.assertTrue(exit_signal)
        
        # Check event emission
        self.mock_event_bus.emit.assert_any_call(
            "strategy.exit", 
            {
                'strategy_id': self.strategy.id,
                'strategy_type': 'volatility',
                'timestamp': unittest.mock.ANY,
                'position_id': self.long_position['id'],
                'instrument': unittest.mock.ANY,
                'price': test_data['close'].iloc[-1],
                'reason': unittest.mock.ANY,
                'position_duration': unittest.mock.ANY,
                'pnl_pct': unittest.mock.ANY,
                'volatility_state': self.strategy._state['volatility_state']
            }
        )

    def test_exit_signal_target_reached(self):
        """Test exit signal generation when target is reached."""
        # Set up data for target reached
        test_data = self.data_with_indicators.copy()
        
        # For short reversal - price below middle band
        test_data.loc[test_data.index[-1], 'close'] = test_data['bb_middle'].iloc[-1] * 0.95
        
        # Check exit signal for short position with reversal entry
        self.short_position['entry_reason'] = 'reversal after volatility expansion'
        
        # Generate exit signal
        exit_signal = self.strategy.exit_signal(test_data, self.short_position)
        
        # Check that exit signal is generated
        self.assertTrue(exit_signal)

    def test_detect_volatility_shift(self):
        """Test volatility shift detection."""
        # Create test data with volatility increase
        atr_increase = pd.Series([0.5] * 20)
        atr_increase.iloc[-1] = 0.9  # 80% increase
        
        # Detect shift
        shift_detected = self.strategy._detect_volatility_shift(self.data, atr_increase)
        
        # Check that shift is detected
        self.assertTrue(shift_detected)
        
        # Create test data with volatility decrease
        atr_decrease = pd.Series([0.5] * 20)
        atr_decrease.iloc[-1] = 0.25  # 50% decrease
        
        # Detect shift
        shift_detected = self.strategy._detect_volatility_shift(self.data, atr_decrease)
        
        # Check that shift is detected
        self.assertTrue(shift_detected)
        
        # Create test data with stable volatility
        atr_stable = pd.Series([0.5] * 20)
        atr_stable.iloc[-1] = 0.51  # 2% increase
        
        # Detect shift
        shift_detected = self.strategy._detect_volatility_shift(self.data, atr_stable)
        
        # Check that no shift is detected
        self.assertFalse(shift_detected)

    def test_adapt_to_regime(self):
        """Test strategy adaptation to market regime."""
        # Create regime data
        regime_data = {
            'id': 'high-vol-uptrend',
            'directional_bias': DirectionalBias.UPWARD,
            'volatility_regime': VolatilityRegime.HIGH,
            'peak_hour': 14  # US market opening
        }
        
        # Store original parameters
        original_threshold = self.strategy._state['volatility_threshold_modifier']
        original_stop_multiplier = self.strategy._state['trailing_stop_multiplier']
        
        # Adapt to regime
        self.strategy.adapt_to_regime(regime_data)
        
        # Check that parameters were updated
        self.assertNotEqual(self.strategy._state['volatility_threshold_modifier'], original_threshold)
        self.assertNotEqual(self.strategy._state['trailing_stop_multiplier'], original_stop_multiplier)
        
        # Check that threshold was increased for high volatility
        self.assertGreater(self.strategy._state['volatility_threshold_modifier'], original_threshold)
        
        # Check event emission
        self.mock_event_bus.emit.assert_called_with(
            "strategy.regime_adaptation", 
            {
                'strategy_id': self.strategy.id,
                'strategy_type': 'volatility',
                'timestamp': unittest.mock.ANY,
                'regime_id': regime_data['id'],
                'parameters': unittest.mock.ANY
            }
        )

    def test_update_parameters_online(self):
        """Test online parameter updates based on performance."""
        # Create performance metrics
        performance_metrics = {
            'sharpe_ratio': 1.2,
            'win_rate': 0.65,
            'profit_factor': 1.8
        }
        
        # Create market conditions
        market_conditions = {
            'volatility': 0.8,  # High volatility
            'volatility_trend': 'increasing'
        }
        
        # Store original parameters
        original_threshold = self.strategy._state['volatility_threshold_modifier']
        original_stop_multiplier = self.strategy._state['trailing_stop_multiplier']
        
        # Update parameters
        self.strategy.update_parameters_online(performance_metrics, market_conditions)
        
        # Check that parameters were updated
        self.assertNotEqual(self.strategy._state['volatility_threshold_modifier'], original_threshold)
        self.assertNotEqual(self.strategy._state['trailing_stop_multiplier'], original_stop_multiplier)
        
        # Check event emission
        self.mock_event_bus.emit.assert_called_with(
            "strategy.online_update", 
            {
                'strategy_id': self.strategy.id,
                'strategy_type': 'volatility',
                'timestamp': unittest.mock.ANY,
                'new_parameters': unittest.mock.ANY,
                'performance_metrics': performance_metrics,
                'market_conditions': market_conditions
            }
        )

    def test_risk_parameters(self):
        """Test risk parameter calculation."""
        # Generate risk parameters
        risk_params = self.strategy.risk_parameters(self.data_with_indicators, 100.0)
        
        # Check that all required parameters are present
        self.assertIn('stop_loss_pct', risk_params)
        self.assertIn('take_profit_pct', risk_params)
        self.assertIn('position_size', risk_params)
        self.assertIn('stop_price', risk_params)
        self.assertIn('take_profit_price', risk_params)
        self.assertIn('atr', risk_params)
        self.assertIn('volatility_state', risk_params)
        
        # Check event emission
        self.mock_event_bus.emit.assert_called_with(
            "strategy.risk_parameters", 
            {
                'strategy_id': self.strategy.id,
                'strategy_type': 'volatility',
                'timestamp': unittest.mock.ANY,
                'entry_price': 100.0,
                'stop_loss_pct': risk_params['stop_loss_pct'],
                'take_profit_pct': risk_params['take_profit_pct'],
                'position_size': risk_params['position_size'],
                'risk_reward_ratio': unittest.mock.ANY,
                'volatility_state': self.strategy._state['volatility_state'],
                'volatility_trend': self.strategy._state['volatility_trend']
            }
        )
        
    def test_get_required_features(self):
        """Test getting required features."""
        # Get required features
        features = self.strategy.get_required_features()
        
        # Check that all required features are present
        self.assertIn('open', features)
        self.assertIn('high', features)
        self.assertIn('low', features)
        self.assertIn('close', features)
        self.assertIn('volume', features)
        self.assertIn('bb_upper', features)
        self.assertIn('bb_lower', features)
        self.assertIn('bb_middle', features)
        self.assertIn('atr_14', features)
        
        # Disable Bollinger Bands
        self.strategy.parameters['use_bbands'] = False
        features = self.strategy.get_required_features()
        
        # Check that Bollinger Band features are not present
        self.assertNotIn('bb_upper', features)
        self.assertNotIn('bb_lower', features)
        self.assertNotIn('bb_middle', features)
        
        # Re-enable for other tests
        self.strategy.parameters['use_bbands'] = True

    def test_volume_filter(self):
        """Test that volume filter works properly."""
        # Set strategy state to contraction for clear signal
        self.strategy._state['volatility_state'] = "contraction"
        
        # Enable volume filter
        self.strategy.parameters['volume_filter'] = True
        
        # Modify data to create a breakout with low volume
        test_data = self.data_with_indicators.copy()
        test_data.loc[test_data.index[-1], 'close'] = test_data['bb_upper'].iloc[-1] * 1.05
        test_data.loc[test_data.index[-1], 'volume'] = test_data['volume'].iloc[-20:].mean() * 0.5  # Very low volume
        
        # Generate signal
        signal = self.strategy.generate_signal(test_data)
        
        # Check that no signal is generated due to volume filter
        self.assertIsNone(signal)
        
        # Now with sufficient volume
        test_data.loc[test_data.index[-1], 'volume'] = test_data['volume'].iloc[-20:].mean() * 1.5  # High volume
        
        # Generate signal
        signal = self.strategy.generate_signal(test_data)
        
        # Check that signal is generated with adequate volume
        self.assertEqual(signal, "long")

    def test_handle_parameter_update(self):
        """Test parameter update handling."""
        # Create parameter update event
        event_data = {
            'parameters': {
                'bb_period': 25,
                'atr_period': 16,
                'expansion_threshold': 1.6
            }
        }
        
        # Handle parameter update
        self.strategy._handle_parameter_update(event_data)
        
        # Check that parameters were updated
        self.assertEqual(self.strategy.parameters['bb_period'], 25)
        self.assertEqual(self.strategy.parameters['atr_period'], 16)
        self.assertEqual(self.strategy.parameters['expansion_threshold'], 1.6)
        
        # Test with invalid parameter
        event_data = {
            'parameters': {
                'invalid_param': 100
            }
        }
        
        # Handle parameter update
        self.strategy._handle_parameter_update(event_data)
        
        # Check that invalid parameter was not added
        self.assertNotIn('invalid_param', self.strategy.parameters)

    def test_handle_regime_change(self):
        """Test regime change handling."""
        # Create regime change event
        event_data = {
            'regime': {
                'id': 'low-vol-range',
                'directional_bias': DirectionalBias.NEUTRAL,
                'volatility_regime': VolatilityRegime.LOW
            },
            'regime_id': 'low-vol-range'
        }
        
        # Mock adapt_to_regime method
        with patch.object(self.strategy, 'adapt_to_regime') as mock_adapt:
            # Handle regime change
            self.strategy._handle_regime_change(event_data)
            
            # Check that adapt_to_regime was called
            mock_adapt.assert_called_with(event_data['regime'])


if __name__ == '__main__':
    unittest.main()