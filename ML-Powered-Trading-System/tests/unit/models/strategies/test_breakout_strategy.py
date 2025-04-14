import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import time

from models.strategies.breakout_strategy import BreakoutStrategy
from models.strategies.strategy_base import DirectionalBias, VolatilityRegime
from core.event_bus import EventBus


class TestBreakoutStrategy(unittest.TestCase):
    """Test cases for the BreakoutStrategy class."""

    def setUp(self):
        """Set up test fixtures before each test method is run."""
        # Default parameters for testing
        self.test_params = {
            'consolidation_period': 10,
            'breakout_threshold': 1.0,
            'volume_confirmation': False,  # Simplified for tests
            'min_volume_increase': 1.5,
            'atr_period': 7,
            'atr_multiplier': 2.0,
            'min_consolidation_length': 5,
            'max_consolidation_width_pct': 0.05,
            'min_consolidation_quality': 0.6,
            'false_breakout_filter': False,  # Simplified for tests
            'confirmation_candles': 1,
            'multi_timeframe_confirmation': False,  # Simplified for tests
            'min_bars': 15
        }
        
        # Initialize strategy
        self.strategy = BreakoutStrategy(
            name="TestBreakout", 
            parameters=self.test_params, 
            strategy_id="test-breakout-1"
        )
        
        # Mock EventBus subscribe method
        self.original_subscribe = EventBus.subscribe
        EventBus.subscribe = MagicMock()
        
        # Mock EventBus emit method
        self.original_emit = EventBus.emit
        EventBus.emit = MagicMock()
        
        # Sample data for testing
        self.create_sample_data()

    def tearDown(self):
        """Tear down test fixtures after each test method is run."""
        # Restore original EventBus methods
        EventBus.subscribe = self.original_subscribe
        EventBus.emit = self.original_emit

    def create_sample_data(self):
        """Create sample data for testing."""
        # Base data with a consolidation pattern
        dates = pd.date_range(start='2023-01-01', periods=30, freq='D')
        
        # Base pattern
        close = np.concatenate([
            np.linspace(100, 101, 10),  # Slight uptrend
            np.linspace(101, 100, 10),  # Slight downtrend
            np.linspace(100, 105, 10)   # Breakout
        ])
        
        # Create high, low based on close with some noise
        high = close + np.random.uniform(0, 0.5, size=close.shape)
        low = close - np.random.uniform(0, 0.5, size=close.shape)
        open_prices = close - np.random.uniform(-0.5, 0.5, size=close.shape)
        
        # Create volume
        volume = np.random.uniform(1000, 2000, size=close.shape)
        volume[-5:] = volume[-5:] * 1.8  # Volume spike during breakout
        
        # Create ATR
        atr = np.random.uniform(0.2, 0.4, size=close.shape)
        
        # Create DataFrame
        self.sample_data = pd.DataFrame({
            'open': open_prices,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume,
            'atr_14': atr
        }, index=dates)
        
        # Create downward breakout sample
        close_down = np.concatenate([
            np.linspace(100, 101, 10),  # Slight uptrend
            np.linspace(101, 100, 10),  # Slight downtrend
            np.linspace(100, 95, 10)    # Downward breakout
        ])
        
        high_down = close_down + np.random.uniform(0, 0.5, size=close_down.shape)
        low_down = close_down - np.random.uniform(0, 0.5, size=close_down.shape)
        open_down = close_down - np.random.uniform(-0.5, 0.5, size=close_down.shape)
        
        self.sample_data_down = pd.DataFrame({
            'open': open_down,
            'high': high_down,
            'low': low_down,
            'close': close_down,
            'volume': volume,
            'atr_14': atr
        }, index=dates)
        
        # Create no breakout sample (consolidation only)
        close_no_breakout = np.concatenate([
            np.linspace(100, 101, 10),    # Slight uptrend
            np.linspace(101, 100, 10),    # Slight downtrend
            np.linspace(100, 100.5, 10)   # No breakout
        ])
        
        high_no_breakout = close_no_breakout + np.random.uniform(0, 0.5, size=close_no_breakout.shape)
        low_no_breakout = close_no_breakout - np.random.uniform(0, 0.5, size=close_no_breakout.shape)
        open_no_breakout = close_no_breakout - np.random.uniform(-0.5, 0.5, size=close_no_breakout.shape)
        
        self.sample_data_no_breakout = pd.DataFrame({
            'open': open_no_breakout,
            'high': high_no_breakout,
            'low': low_no_breakout,
            'close': close_no_breakout,
            'volume': volume,
            'atr_14': atr
        }, index=dates)
        
        # Create invalid data (missing columns)
        self.invalid_data = pd.DataFrame({
            'close': close,
            'volume': volume
        }, index=dates)

    def test_initialization(self):
        """Test that the strategy initializes correctly with default and custom parameters."""
        # Test with default parameters
        strategy_default = BreakoutStrategy()
        self.assertEqual(strategy_default.name, "Breakout")
        self.assertEqual(strategy_default.parameters['consolidation_period'], 20)
        
        # Test with custom parameters
        self.assertEqual(self.strategy.name, "TestBreakout")
        self.assertEqual(self.strategy.parameters['consolidation_period'], 10)
        self.assertEqual(self.strategy.parameters['breakout_threshold'], 1.0)
        
        # Verify event bus registration
        EventBus.subscribe.assert_any_call(f"strategy.{self.strategy.id}.parameter_update", 
                                           self.strategy._handle_parameter_update)
        EventBus.subscribe.assert_any_call("market.regime_change", 
                                           self.strategy._handle_regime_change)

    def test_validate_parameters(self):
        """Test parameter validation logic."""
        # Test valid parameters (should not raise exception)
        self.strategy._validate_parameters()
        
        # Test invalid period
        with self.assertRaises(ValueError):
            invalid_strategy = BreakoutStrategy(parameters={'consolidation_period': 0})
            
        # Test invalid threshold
        with self.assertRaises(ValueError):
            invalid_strategy = BreakoutStrategy(parameters={'breakout_threshold': -1.0})
            
        # Test invalid quality threshold
        with self.assertRaises(ValueError):
            invalid_strategy = BreakoutStrategy(parameters={'min_consolidation_quality': 1.5})

    def test_validate_data(self):
        """Test data validation logic."""
        # Valid data should return True
        self.assertTrue(self.strategy.validate_data(self.sample_data))
        
        # Empty DataFrame should raise ValueError
        with self.assertRaises(ValueError):
            self.strategy.validate_data(pd.DataFrame())
            
        # Missing required columns should raise ValueError
        with self.assertRaises(ValueError):
            self.strategy.validate_data(self.invalid_data)

    def test_detect_consolidation(self):
        """Test consolidation detection logic."""
        # Test with valid consolidation
        consolidation = self.strategy._detect_consolidation(self.sample_data)
        self.assertTrue(consolidation['is_valid'])
        self.assertIn('support', consolidation)
        self.assertIn('resistance', consolidation)
        self.assertIn('quality', consolidation)
        
        # Test with insufficient data
        insufficient_data = self.sample_data.iloc[:5]
        consolidation_insufficient = self.strategy._detect_consolidation(insufficient_data)
        self.assertFalse(consolidation_insufficient['is_valid'])
        
        # Test with too wide range
        wide_range_strategy = BreakoutStrategy(parameters={'max_consolidation_width_pct': 0.01})
        consolidation_wide = wide_range_strategy._detect_consolidation(self.sample_data)
        self.assertFalse(consolidation_wide['is_valid'])

    def test_detect_breakout(self):
        """Test breakout detection logic."""
        # First detect consolidation
        consolidation = self.strategy._detect_consolidation(self.sample_data)
        
        # Test with upward breakout
        breakout = self.strategy._detect_breakout(self.sample_data, consolidation)
        self.assertTrue(breakout['is_breakout'])
        self.assertEqual(breakout['direction'], 'up')
        self.assertGreater(breakout['strength'], 0)
        
        # Test with downward breakout
        consolidation_down = self.strategy._detect_consolidation(self.sample_data_down)
        breakout_down = self.strategy._detect_breakout(self.sample_data_down, consolidation_down)
        self.assertTrue(breakout_down['is_breakout'])
        self.assertEqual(breakout_down['direction'], 'down')
        self.assertGreater(breakout_down['strength'], 0)
        
        # Test with no breakout
        consolidation_no = self.strategy._detect_consolidation(self.sample_data_no_breakout)
        breakout_no = self.strategy._detect_breakout(self.sample_data_no_breakout, consolidation_no)
        self.assertFalse(breakout_no['is_breakout'])
        
        # Test with invalid consolidation
        invalid_consolidation = {'is_valid': False}
        breakout_invalid = self.strategy._detect_breakout(self.sample_data, invalid_consolidation)
        self.assertFalse(breakout_invalid['is_breakout'])

    def test_generate_signal(self):
        """Test signal generation."""
        # Test with upward breakout
        signal = self.strategy.generate_signal(self.sample_data)
        self.assertEqual(signal, 'long')
        
        # Test with downward breakout
        signal_down = self.strategy.generate_signal(self.sample_data_down)
        self.assertEqual(signal_down, 'short')
        
        # Test with no breakout
        signal_no = self.strategy.generate_signal(self.sample_data_no_breakout)
        self.assertIsNone(signal_no)
        
        # Test with insufficient data
        insufficient_data = self.sample_data.iloc[:10]
        signal_insufficient = self.strategy.generate_signal(insufficient_data)
        self.assertIsNone(signal_insufficient)
        
        # Test with invalid data
        signal_invalid = self.strategy.generate_signal(self.invalid_data)
        self.assertIsNone(signal_invalid)
        
        # Verify signal event emission
        EventBus.emit.assert_any_call("strategy.signal", any)

    def test_apply_directional_bias(self):
        """Test directional bias application to signals."""
        # Set directional bias to upward
        self.strategy._state['directional_bias'] = DirectionalBias.UPWARD
        
        # Test with upward breakout (should keep signal)
        signal = self.strategy._apply_directional_bias('long', self.sample_data)
        self.assertEqual(signal, 'long')
        
        # Test with downward breakout and strong signal
        # Mock the breakout detection to return strong signal
        with patch.object(self.strategy, '_detect_breakout') as mock_detect:
            mock_detect.return_value = {
                'is_breakout': True,
                'direction': 'down',
                'strength': 2.0  # Strong signal
            }
            signal_down = self.strategy._apply_directional_bias('short', self.sample_data)
            self.assertEqual(signal_down, 'short')
        
        # Test with downward breakout and weak signal
        with patch.object(self.strategy, '_detect_breakout') as mock_detect:
            mock_detect.return_value = {
                'is_breakout': True,
                'direction': 'down',
                'strength': 1.0  # Weak signal
            }
            signal_down_weak = self.strategy._apply_directional_bias('short', self.sample_data)
            self.assertIsNone(signal_down_weak)  # Signal should be filtered out
        
        # Set directional bias to downward
        self.strategy._state['directional_bias'] = DirectionalBias.DOWNWARD
        
        # Test with downward breakout (should keep signal)
        signal_down = self.strategy._apply_directional_bias('short', self.sample_data_down)
        self.assertEqual(signal_down, 'short')
        
        # Test with upward breakout and weak signal
        with patch.object(self.strategy, '_detect_breakout') as mock_detect:
            mock_detect.return_value = {
                'is_breakout': True,
                'direction': 'up',
                'strength': 1.0  # Weak signal
            }
            signal_up_weak = self.strategy._apply_directional_bias('long', self.sample_data)
            self.assertIsNone(signal_up_weak)  # Signal should be filtered out
        
        # Reset directional bias to neutral
        self.strategy._state['directional_bias'] = DirectionalBias.NEUTRAL

    def test_risk_parameters(self):
        """Test risk parameter calculation."""
        # Add a detected pattern to state
        self.strategy._state['detected_patterns'] = [{
            'type': 'support_resistance',
            'support': 95.0,
            'resistance': 105.0,
            'quality': 0.8,
            'timestamp': time.time()
        }]
        
        # Test risk parameters for a long position
        entry_price = 105.0
        risk_params = self.strategy.risk_parameters(self.sample_data, entry_price)
        
        self.assertIn('stop_loss_pct', risk_params)
        self.assertIn('take_profit_pct', risk_params)
        self.assertIn('position_size', risk_params)
        self.assertIn('stop_price', risk_params)
        self.assertIn('take_profit_price', risk_params)
        self.assertIn('atr', risk_params)
        
        # Verify stop loss calculation
        self.assertGreater(risk_params['stop_loss_pct'], 0)
        self.assertLess(risk_params['stop_price'], entry_price)
        
        # Verify take profit calculation
        self.assertGreater(risk_params['take_profit_pct'], risk_params['stop_loss_pct'])
        self.assertGreater(risk_params['take_profit_price'], entry_price)
        
        # Test with empty data
        risk_params_empty = self.strategy.risk_parameters(pd.DataFrame(), entry_price)
        self.assertEqual(risk_params_empty['stop_loss_pct'], 0.01)  # Default fallback
        
        # Test with zero entry price
        risk_params_zero = self.strategy.risk_parameters(self.sample_data, 0)
        self.assertEqual(risk_params_zero['stop_loss_pct'], 0.01)  # Default fallback
        
        # Verify risk event emission
        EventBus.emit.assert_any_call("strategy.risk_parameters", any)

    def test_exit_signal(self):
        """Test exit signal generation."""
        # Create a long position
        position = {
            'entry_price': 102.0,
            'direction': 'long',
            'breakout_level': 101.0,
            'consolidation_low': 98.0,
            'consolidation_high': 103.0,
            'entry_time': time.time() - 3600  # 1 hour ago
        }
        
        # Test normal case (no exit)
        exit_signal = self.strategy.exit_signal(self.sample_data, position)
        self.assertFalse(exit_signal)
        
        # Test retracement into consolidation zone
        # Modify the last price to simulate retracement
        retracement_data = self.sample_data.copy()
        retracement_data.iloc[-1, retracement_data.columns.get_loc('close')] = 99.0  # Below retracement level
        
        exit_signal_retracement = self.strategy.exit_signal(retracement_data, position)
        self.assertTrue(exit_signal_retracement)
        
        # Test volatility spike
        volatility_data = self.sample_data.copy()
        # Increase ATR to simulate volatility spike
        volatility_data.iloc[-1, volatility_data.columns.get_loc('atr_14')] = 2.0  
        
        exit_signal_volatility = self.strategy.exit_signal(volatility_data, position)
        self.assertTrue(exit_signal_volatility)
        
        # Test volume spike on price decline for long position
        volume_data = self.sample_data.copy()
        volume_data.iloc[-1, volume_data.columns.get_loc('close')] = 100.5  # Price decline
        volume_data.iloc[-1, volume_data.columns.get_loc('volume')] = 5000  # Volume spike
        
        exit_signal_volume = self.strategy.exit_signal(volume_data, position)
        self.assertTrue(exit_signal_volume)
        
        # Test with empty data
        exit_signal_empty = self.strategy.exit_signal(pd.DataFrame(), position)
        self.assertFalse(exit_signal_empty)
        
        # Verify exit event emission
        EventBus.emit.assert_any_call("strategy.exit", any)

    def test_handle_parameter_update(self):
        """Test parameter update handling."""
        # Mock the _validate_parameters method
        with patch.object(self.strategy, '_validate_parameters') as mock_validate:
            # Create update event
            event_data = {
                'parameters': {
                    'consolidation_period': 15,
                    'breakout_threshold': 1.2
                }
            }
            
            # Handle the update
            self.strategy._handle_parameter_update(event_data)
            
            # Verify parameters were updated
            self.assertEqual(self.strategy.parameters['consolidation_period'], 15)
            self.assertEqual(self.strategy.parameters['breakout_threshold'], 1.2)
            
            # Verify validation was called
            mock_validate.assert_called_once()

    def test_handle_regime_change(self):
        """Test regime change handling."""
        # Mock the adapt_to_regime method
        with patch.object(self.strategy, 'adapt_to_regime') as mock_adapt:
            # Create regime change event
            event_data = {
                'regime': 'high_volatility',
                'regime_id': 'volatile_market'
            }
            
            # Handle the regime change
            self.strategy._handle_regime_change(event_data)
            
            # Verify adapt_to_regime was called
            mock_adapt.assert_called_once_with('high_volatility')

    def test_record_latency(self):
        """Test latency recording."""
        # Record a latency metric
        start_time = time.time() - 0.1  # 100ms ago
        self.strategy._record_latency(start_time, "test_stage", "test_signal")
        
        # Verify metric was recorded
        metrics = self.strategy._state['latency_metrics']
        self.assertGreaterEqual(len(metrics), 1)
        last_metric = metrics[-1]
        
        self.assertEqual(last_metric['stage'], "test_stage")
        self.assertEqual(last_metric['signal'], "test_signal")
        self.assertGreaterEqual(last_metric['latency'], 0.0)
        
        # Verify performance metric was emitted
        EventBus.emit.assert_any_call(
            f"strategy.{self.strategy.id}.latency", 
            any, 
            tags={'stage': "test_stage", 'signal': "test_signal"}
        )

    def test_base_generate_signal(self):
        """Test base signal generation logic."""
        # Mock internal methods to test the base_generate_signal independently
        with patch.object(self.strategy, '_detect_consolidation') as mock_consolidation, \
             patch.object(self.strategy, '_detect_breakout') as mock_breakout, \
             patch.object(self.strategy, '_is_false_breakout') as mock_false_breakout, \
             patch.object(self.strategy, '_confirm_volume') as mock_confirm_volume, \
             patch.object(self.strategy, '_confirm_multi_timeframe') as mock_multi_timeframe, \
             patch.object(self.strategy, 'log_signal') as mock_log_signal, \
             patch.object(self.strategy, '_emit_signal_event') as mock_emit:
            
            # Setup mocks
            mock_consolidation.return_value = {'is_valid': True}
            mock_breakout.return_value = {
                'is_breakout': True, 
                'direction': 'up', 
                'strength': 1.5
            }
            mock_false_breakout.return_value = False
            mock_confirm_volume.return_value = True
            mock_multi_timeframe.return_value = True
            
            # Run the method
            signal = self.strategy._base_generate_signal(self.sample_data, time.time())
            
            # Verify signal and method calls
            self.assertEqual(signal, 'long')
            mock_consolidation.assert_called_once()
            mock_breakout.assert_called_once()
            mock_log_signal.assert_called_once()
            mock_emit.assert_called_once()
            
            # Reset mocks for downward test
            mock_consolidation.reset_mock()
            mock_breakout.reset_mock()
            mock_log_signal.reset_mock()
            mock_emit.reset_mock()
            
            # Setup for downward breakout
            mock_breakout.return_value = {
                'is_breakout': True, 
                'direction': 'down', 
                'strength': 1.5
            }
            
            # Run again
            signal = self.strategy._base_generate_signal(self.sample_data, time.time())
            
            # Verify signal and method calls
            self.assertEqual(signal, 'short')
            mock_log_signal.assert_called_once()
            mock_emit.assert_called_once()
            
            # Reset mocks for no breakout test
            mock_consolidation.reset_mock()
            mock_breakout.reset_mock()
            
            # Setup for no breakout
            mock_breakout.return_value = {'is_breakout': False}
            
            # Run again
            signal = self.strategy._base_generate_signal(self.sample_data, time.time())
            
            # Verify no signal
            self.assertIsNone(signal)
            
            # Reset mocks for no consolidation test
            mock_consolidation.reset_mock()
            
            # Setup for no consolidation
            mock_consolidation.return_value = {'is_valid': False}
            
            # Run again
            signal = self.strategy._base_generate_signal(self.sample_data, time.time())
            
            # Verify no signal
            self.assertIsNone(signal)

    def test_emit_signal_event(self):
        """Test signal event emission."""
        # Setup test data
        breakout_info = {
            'direction': 'up',
            'strength': 1.5
        }
        
        # Call the method
        self.strategy._emit_signal_event('long', self.sample_data, breakout_info)
        
        # Verify event emission
        EventBus.emit.assert_called_once_with(
            "strategy.signal", 
            {
                'strategy_id': self.strategy.id,
                'strategy_type': 'breakout',
                'timestamp': any,
                'signal': 'long',
                'instrument': 'unknown',
                'price': any,
                'strength': 1.5,
                'confidence': any,
                'metadata': any
            }
        )

    def test_emit_risk_event(self):
        """Test risk event emission."""
        # Call the method
        self.strategy._emit_risk_event(100.0, 0.02, 0.06, 1.0)
        
        # Verify event emission
        EventBus.emit.assert_called_once_with(
            "strategy.risk_parameters", 
            {
                'strategy_id': self.strategy.id,
                'strategy_type': 'breakout',
                'timestamp': any,
                'entry_price': 100.0,
                'stop_loss_pct': 0.02,
                'take_profit_pct': 0.06,
                'position_size': 1.0,
                'risk_reward_ratio': 3.0
            }
        )

    def test_emit_exit_event(self):
        """Test exit event emission."""
        # Setup position info
        position = {
            'id': 'pos123',
            'entry_price': 100.0,
            'direction': 'long',
            'entry_time': time.time() - 3600  # 1 hour ago
        }
        
        # Call the method
        self.strategy._emit_exit_event(self.sample_data, position, "Test exit reason")
        
        # Verify event emission
        EventBus.emit.assert_called_once_with(
            "strategy.exit", 
            {
                'strategy_id': self.strategy.id,
                'strategy_type': 'breakout',
                'timestamp': any,
                'position_id': 'pos123',
                'instrument': 'unknown',
                'price': any,
                'reason': "Test exit reason",
                'position_duration': any,
                'pnl_pct': any
            }
        )

    def test_edge_cases(self):
        """Test various edge cases."""
        # Test with None data
        signal_none = self.strategy.generate_signal(None)
        self.assertIsNone(signal_none)
        
        # Test with single row data
        single_row = self.sample_data.iloc[0:1]
        signal_single = self.strategy.generate_signal(single_row)
        self.assertIsNone(signal_single)
        
        # Test exit signal with missing position data
        incomplete_position = {'direction': 'long'}
        exit_signal = self.strategy.exit_signal(self.sample_data, incomplete_position)
        self.assertFalse(exit_signal)
        
        # Test risk parameters with no detected patterns
        self.strategy._state['detected_patterns'] = []
        risk_params = self.strategy.risk_parameters(self.sample_data, 100.0)
        self.assertGreater(risk_params['stop_loss_pct'], 0)


if __name__ == '__main__':
    unittest.main()