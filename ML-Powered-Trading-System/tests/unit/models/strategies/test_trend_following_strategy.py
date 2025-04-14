import unittest
import pandas as pd
import numpy as np
import time
from unittest.mock import MagicMock, patch

# Import the strategy to test
from models.strategies.trend_following_strategy import TrendFollowingStrategy
from models.strategies.strategy_base import DirectionalBias, VolatilityRegime


class TestTrendFollowingStrategy(unittest.TestCase):
    """Test cases for TrendFollowingStrategy."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Default parameters for testing
        self.params = {
            'fast_ma_period': 10,
            'slow_ma_period': 30,
            'adx_period': 14,
            'adx_threshold': 25,
            'atr_period': 14,
            'min_bars': 50
        }
        
        # Create strategy instance
        self.strategy = TrendFollowingStrategy(name="TestTrend", parameters=self.params, strategy_id="test-trend-123")
        
        # Create sample data
        self.create_sample_data()
        
        # Mock EventBus
        self.event_bus_patcher = patch('models.strategies.trend_following_strategy.EventBus')
        self.mock_event_bus = self.event_bus_patcher.start()
        
        # Mock PerformanceMetrics
        self.perf_metrics_patcher = patch('models.strategies.trend_following_strategy.PerformanceMetrics')
        self.mock_perf_metrics = self.perf_metrics_patcher.start()

    def tearDown(self):
        """Tear down test fixtures after each test method."""
        # Stop patches
        self.event_bus_patcher.stop()
        self.perf_metrics_patcher.stop()

    def create_sample_data(self):
        """Create sample market data for testing."""
        # Generate date range
        dates = pd.date_range(start='2023-01-01', periods=200, freq='D')
        
        # Create uptrend price series
        close_uptrend = np.linspace(100, 200, 200) + np.random.normal(0, 5, 200)
        
        # Create downtrend price series
        close_downtrend = np.linspace(200, 100, 200) + np.random.normal(0, 5, 200)
        
        # Create range-bound price series
        close_range = np.sin(np.linspace(0, 6*np.pi, 200)) * 20 + 150 + np.random.normal(0, 2, 200)
        
        # Create OHLCV data
        self.uptrend_data = pd.DataFrame({
            'date': dates,
            'open': close_uptrend - 2,
            'high': close_uptrend + 5,
            'low': close_uptrend - 5,
            'close': close_uptrend,
            'volume': np.random.randint(1000, 10000, 200)
        }).set_index('date')
        
        self.downtrend_data = pd.DataFrame({
            'date': dates,
            'open': close_downtrend + 2,
            'high': close_downtrend + 5,
            'low': close_downtrend - 5,
            'close': close_downtrend,
            'volume': np.random.randint(1000, 10000, 200)
        }).set_index('date')
        
        self.range_data = pd.DataFrame({
            'date': dates,
            'open': close_range - 2,
            'high': close_range + 5,
            'low': close_range - 5,
            'close': close_range,
            'volume': np.random.randint(1000, 10000, 200)
        }).set_index('date')

        # Pre-calculate indicators for testing
        for df in [self.uptrend_data, self.downtrend_data, self.range_data]:
            df['ADX'] = self.strategy._calculate_adx(df)
            df['atr_14'] = self.strategy._calculate_atr(df)

    def test_initialization(self):
        """Test strategy initialization with parameters."""
        strategy = TrendFollowingStrategy(
            name="CustomTrend", 
            parameters={'fast_ma_period': 15, 'slow_ma_period': 45}, 
            strategy_id="custom-123"
        )
        
        self.assertEqual(strategy.name, "CustomTrend")
        self.assertEqual(strategy.id, "custom-123")
        self.assertEqual(strategy.parameters['fast_ma_period'], 15)
        self.assertEqual(strategy.parameters['slow_ma_period'], 45)
        self.assertEqual(strategy.trend_filter_strength, 1.0)  # Default value

    def test_parameter_validation(self):
        """Test parameter validation."""
        # Test invalid parameters
        with self.assertRaises(ValueError):
            TrendFollowingStrategy(parameters={'fast_ma_period': 50, 'slow_ma_period': 20})
        
        with self.assertRaises(ValueError):
            TrendFollowingStrategy(parameters={'fast_ma_period': -5, 'slow_ma_period': 20})
        
        with self.assertRaises(ValueError):
            TrendFollowingStrategy(parameters={'fast_ma_period': 10, 'slow_ma_period': 20, 'adx_threshold': -5})

    def test_data_validation(self):
        """Test data validation."""
        # Test missing columns
        invalid_data = pd.DataFrame({
            'open': [100, 101, 102],
            'close': [105, 106, 107]
            # Missing high, low, volume
        })
        
        with self.assertRaises(ValueError):
            self.strategy.validate_data(invalid_data)
        
        # Test NaN values
        invalid_data = self.uptrend_data.copy()
        invalid_data.iloc[5, 0] = np.nan  # Set a NaN value
        
        with self.assertRaises(ValueError):
            self.strategy.validate_data(invalid_data)
        
        # Test insufficient data
        insufficient_data = self.uptrend_data.iloc[:20]  # Only 20 rows
        
        with self.assertRaises(ValueError):
            self.strategy.validate_data(insufficient_data)

    def test_calculate_indicators(self):
        """Test technical indicator calculations."""
        data = self.uptrend_data.iloc[:50].copy()
        
        # Test ADX calculation
        adx = self.strategy._calculate_adx(data)
        self.assertEqual(len(adx), len(data))
        self.assertTrue(all(0 <= x <= 100 for x in adx.dropna()))
        
        # Test ATR calculation
        atr = self.strategy._calculate_atr(data)
        self.assertEqual(len(atr), len(data))
        self.assertTrue(all(x >= 0 for x in atr.dropna()))

    def test_uptrend_signals(self):
        """Test signal generation in uptrend."""
        # Get segment where crossover should occur
        uptrend_segment = self.uptrend_data.iloc[40:60].copy()  # Adjust range to capture crossover
        
        # Set ADX to strong trend
        uptrend_segment['ADX'] = 35  # Strong trend
        
        # Replace the actual generate_signal call to avoid EventBus/metrics dependencies
        with patch.object(self.strategy, '_emit_signal_event'), \
             patch.object(self.strategy, 'log_signal'), \
             patch.object(self.strategy, '_record_latency'):
            
            # Test strategy for each row in the segment
            signals = []
            for i in range(len(uptrend_segment) - 1):
                data_slice = uptrend_segment.iloc[:i+2]  # Include at least 2 rows
                if len(data_slice) >= self.params['min_bars']:
                    signal = self.strategy.generate_signal(data_slice)
                    signals.append(signal)
            
            # Verify we get at least one long signal in an uptrend
            self.assertIn('long', signals)
            # Verify we don't get short signals in an uptrend
            self.assertNotIn('short', signals)

    def test_downtrend_signals(self):
        """Test signal generation in downtrend."""
        # Get segment where crossover should occur
        downtrend_segment = self.downtrend_data.iloc[40:60].copy()
        
        # Set ADX to strong trend
        downtrend_segment['ADX'] = 35  # Strong trend
        
        # Replace the actual generate_signal call to avoid EventBus/metrics dependencies
        with patch.object(self.strategy, '_emit_signal_event'), \
             patch.object(self.strategy, 'log_signal'), \
             patch.object(self.strategy, '_record_latency'):
            
            # Test strategy for each row in the segment
            signals = []
            for i in range(len(downtrend_segment) - 1):
                data_slice = downtrend_segment.iloc[:i+2]
                if len(data_slice) >= self.params['min_bars']:
                    signal = self.strategy.generate_signal(data_slice)
                    signals.append(signal)
            
            # Verify we get at least one short signal in a downtrend
            self.assertIn('short', signals)
            # Verify we don't get long signals in a downtrend
            self.assertNotIn('long', signals)

    def test_exit_signal(self):
        """Test exit signal generation."""
        data = self.uptrend_data.copy()
        
        # Create a long position
        position = {
            'direction': 'long',
            'entry_price': 120,
            'entry_time': time.time() - 3600  # 1 hour ago
        }
        
        # Test normal case (no exit)
        with patch.object(self.strategy, '_emit_exit_event'), \
             patch.object(self.strategy, 'log_signal'), \
             patch.object(self.strategy, '_record_latency'):
            
            # In uptrend, should not exit long position
            should_exit = self.strategy.exit_signal(data, position)
            self.assertFalse(should_exit)
            
            # Create a case where MA crosses (exit condition)
            modified_data = data.copy()
            # Modify close prices to create MA crossover
            modified_data['close'] = modified_data['close'] * 0.8  # Lower the price
            should_exit = self.strategy.exit_signal(modified_data, position)
            self.assertTrue(should_exit)
            
            # Test exit on ADX weakening
            modified_data = data.copy()
            modified_data['ADX'] = 12  # Below threshold
            should_exit = self.strategy.exit_signal(modified_data, position)
            self.assertTrue(should_exit)

    def test_risk_parameters(self):
        """Test risk parameter calculation."""
        data = self.uptrend_data.copy()
        entry_price = 150.0
        
        # Set up regime characteristics
        self.strategy.regime_characteristics = {
            'volatility_zscore': 0.5,  # Moderate volatility
            'volatility_regime': VolatilityRegime.NORMAL
        }
        
        # Test normal risk parameter calculation
        with patch.object(self.strategy, '_emit_risk_event'), \
             patch.object(self.strategy, '_record_latency'):
            
            risk_params = self.strategy.risk_parameters(data, entry_price)
            
            # Verify required keys exist
            self.assertIn('stop_loss_pct', risk_params)
            self.assertIn('take_profit_pct', risk_params)
            self.assertIn('position_size', risk_params)
            self.assertIn('stop_price', risk_params)
            self.assertIn('take_profit_price', risk_params)
            
            # Verify reasonable values
            self.assertTrue(0 < risk_params['stop_loss_pct'] < 0.1)  # 0-10% stop loss
            self.assertTrue(0 < risk_params['take_profit_pct'] < 0.2)  # 0-20% take profit
            self.assertTrue(0 < risk_params['position_size'] <= 0.05)  # 0-5% position size

    def test_adapt_to_regime(self):
        """Test adaptation to market regime."""
        # Store original parameters
        original_fast, original_slow = self.strategy._state['adaptive_lookbacks']
        original_stop, original_target = self.strategy._state['risk_multipliers']
        
        # Test adaptation to different regimes
        regime_data = {
            'id': 'test-regime-1',
            'directional_bias': DirectionalBias.UPWARD,
            'volatility_regime': VolatilityRegime.HIGH,
            'peak_hour': 11
        }
        
        self.strategy.adapt_to_regime(regime_data)
        
        # Verify parameter changes
        new_fast, new_slow = self.strategy._state['adaptive_lookbacks']
        new_stop, new_target = self.strategy._state['risk_multipliers']
        
        # Parameters should have changed
        self.assertNotEqual((original_fast, original_slow), (new_fast, new_slow))
        self.assertNotEqual((original_stop, original_target), (new_stop, new_target))
        
        # Verify biased signal generation
        # The generate_signal should be modified for upward bias
        self.assertNotEqual(self.strategy.generate_signal, self.strategy.original_generate_signal)
        
        # Reset to neutral bias
        regime_data = {
            'id': 'test-regime-2',
            'directional_bias': DirectionalBias.NEUTRAL,
            'volatility_regime': VolatilityRegime.NORMAL
        }
        
        self.strategy.adapt_to_regime(regime_data)
        
        # Verify signal generation function is reset
        self.assertEqual(self.strategy.generate_signal, self.strategy.original_generate_signal)

    def test_on_trade_completed(self):
        """Test parameter adaptation after trade completion."""
        # Store original parameters
        original_filter_strength = self.strategy.trend_filter_strength
        original_stop, original_target = self.strategy._state['risk_multipliers']
        
        # Test successful trade adaptation
        trade_result = {
            'id': 'test-trade-1',
            'pnl': 100,
            'pnl_pct': 0.06,  # 6% profit
            'regime_id': 'test-regime-1'
        }
        
        with patch.object(self.strategy, '_emit_parameter_adaptation_event'):
            self.strategy.on_trade_completed(trade_result)
        
        # Verify parameter adjustments
        self.assertLess(self.strategy.trend_filter_strength, original_filter_strength)
        new_stop, new_target = self.strategy._state['risk_multipliers']
        self.assertGreater(new_target, original_target)  # Take profit increased
        
        # Test unsuccessful trade adaptation
        original_filter_strength = self.strategy.trend_filter_strength
        original_stop, original_target = self.strategy._state['risk_multipliers']
        
        trade_result = {
            'id': 'test-trade-2',
            'pnl': -50,
            'pnl_pct': -0.04,  # 4% loss
            'regime_id': 'test-regime-2'
        }
        
        with patch.object(self.strategy, '_emit_parameter_adaptation_event'):
            self.strategy.on_trade_completed(trade_result)
        
        # Verify parameter adjustments
        self.assertGreater(self.strategy.trend_filter_strength, original_filter_strength)
        new_stop, new_target = self.strategy._state['risk_multipliers']
        self.assertLess(new_stop, original_stop)  # Stop loss decreased (tightened)

    def test_update_parameters_online(self):
        """Test online parameter updates based on performance metrics."""
        # Store original parameters
        original_fast, original_slow = self.strategy._state['adaptive_lookbacks']
        original_stop, original_target = self.strategy._state['risk_multipliers']
        original_filter_strength = self.strategy.trend_filter_strength
        
        # Test parameter updates with good performance
        performance_metrics = {
            'sharpe_ratio': 1.5,
            'win_rate': 0.65,
            'profit_factor': 1.8
        }
        
        market_conditions = {
            'adx': 35,  # Strong trend
            'volatility': 0.8  # High volatility
        }
        
        with patch.object(self.strategy, '_emit_exit_event'), \
             patch.object(self.strategy, 'log_signal'):
            
            self.strategy.update_parameters_online(performance_metrics, market_conditions)
        
        # Verify parameter adjustments for good performance and strong trend
        new_fast, new_slow = self.strategy._state['adaptive_lookbacks']
        self.assertLess(new_fast, original_fast)  # Shorter MA periods for strong trend
        self.assertLess(new_slow, original_slow)
        
        # Test with poor performance
        original_fast, original_slow = self.strategy._state['adaptive_lookbacks']
        original_filter_strength = self.strategy.trend_filter_strength
        
        performance_metrics = {
            'sharpe_ratio': 0.3,
            'win_rate': 0.35,
            'profit_factor': 0.9
        }
        
        market_conditions = {
            'adx': 12,  # Weak trend
            'volatility': 0.25  # Low volatility
        }
        
        with patch.object(self.strategy, '_emit_exit_event'), \
             patch.object(self.strategy, 'log_signal'):
            
            self.strategy.update_parameters_online(performance_metrics, market_conditions)
        
        # Verify parameter adjustments for poor performance and weak trend
        new_fast, new_slow = self.strategy._state['adaptive_lookbacks']
        new_filter_strength = self.strategy.trend_filter_strength
        
        self.assertGreater(new_fast, original_fast)  # Longer MA periods for weak trend
        self.assertGreater(new_slow, original_slow)
        self.assertGreater(new_filter_strength, original_filter_strength)  # Stricter filter

    def test_dynamic_position_size(self):
        """Test dynamic position sizing."""
        # Set up test conditions
        self.strategy.account_balance = 10000.0
        
        # Default case
        self.strategy.trend_filter_strength = 1.0
        self.strategy.regime_characteristics = {
            'volatility_regime': VolatilityRegime.NORMAL
        }
        
        position_size = self.strategy.dynamic_position_size(self.strategy.account_balance)
        self.assertEqual(position_size, 0.02)  # Default 2%
        
        # Conservative case (high filter strength)
        self.strategy.trend_filter_strength = 1.3
        position_size = self.strategy.dynamic_position_size(self.strategy.account_balance)
        self.assertGreater(position_size, 0.02)  # Should be higher than base
        
        # High volatility case
        self.strategy.trend_filter_strength = 1.0
        self.strategy.regime_characteristics = {
            'volatility_regime': VolatilityRegime.HIGH
        }
        position_size = self.strategy.dynamic_position_size(self.strategy.account_balance)
        self.assertLess(position_size, 0.02)  # Should be lower due to volatility
        
        # Low volatility case
        self.strategy.regime_characteristics = {
            'volatility_regime': VolatilityRegime.LOW
        }
        position_size = self.strategy.dynamic_position_size(self.strategy.account_balance)
        self.assertGreater(position_size, 0.02)  # Should be higher due to low volatility
        
        # Ensure position size is capped
        self.strategy.trend_filter_strength = 0.7
        self.strategy.regime_characteristics = {
            'volatility_regime': VolatilityRegime.LOW
        }
        position_size = self.strategy.dynamic_position_size(self.strategy.account_balance)
        self.assertLessEqual(position_size, 0.05)  # Should be capped at 5%

    def test_cluster_fit(self):
        """Test cluster fitness calculation."""
        # Strong trend, high ADX - should be good fit for trend following
        cluster_metrics_good = {
            'cluster_id': 'cluster-1',
            'ADX_mean': 35,
            'trend_strength': 0.8,
            'directional_bias': 'upward',
            'autocorrelation': 0.6
        }
        
        fitness_good = self.strategy.cluster_fit(cluster_metrics_good)
        self.assertGreater(fitness_good, 0.7)  # Should be high fitness
        
        # Weak trend, low ADX - should be poor fit for trend following
        cluster_metrics_poor = {
            'cluster_id': 'cluster-2',
            'ADX_mean': 10,
            'trend_strength': 0.3,
            'directional_bias': 'neutral',
            'autocorrelation': -0.2
        }
        
        fitness_poor = self.strategy.cluster_fit(cluster_metrics_poor)
        self.assertLess(fitness_poor, 0.5)  # Should be low fitness
        
        # Edge case - missing metrics
        fitness_neutral = self.strategy.cluster_fit({})
        self.assertEqual(fitness_neutral, 0.5)  # Should be neutral when data missing

    def test_get_required_features(self):
        """Test getting required features."""
        features = self.strategy.get_required_features()
        
        # Verify all required features are present
        self.assertIn('open', features)
        self.assertIn('high', features)
        self.assertIn('low', features)
        self.assertIn('close', features)
        self.assertIn('volume', features)
        self.assertIn('adx_14', features)
        self.assertIn('atr_14', features)

    def test_handle_parameter_update(self):
        """Test handling parameter updates from event bus."""
        # Store original parameters
        original_fast = self.strategy.parameters['fast_ma_period']
        original_slow = self.strategy.parameters['slow_ma_period']
        
        # Create parameter update event
        event_data = {
            'parameters': {
                'fast_ma_period': original_fast + 5,
                'slow_ma_period': original_slow + 10
            }
        }
        
        # Mock _validate_parameters to avoid validation issues
        with patch.object(self.strategy, '_validate_parameters'):
            self.strategy._handle_parameter_update(event_data)
        
        # Verify parameters were updated
        self.assertEqual(self.strategy.parameters['fast_ma_period'], original_fast + 5)
        self.assertEqual(self.strategy.parameters['slow_ma_period'], original_slow + 10)

    def test_handle_regime_change(self):
        """Test handling regime change events."""
        # Mock adapt_to_regime to verify it's called
        with patch.object(self.strategy, 'adapt_to_regime') as mock_adapt:
            event_data = {
                'regime_id': 'new-regime-1',
                'regime': {'id': 'new-regime-1', 'directional_bias': DirectionalBias.UPWARD}
            }
            
            self.strategy._handle_regime_change(event_data)
            
            # Verify adapt_to_regime was called with the regime data
            mock_adapt.assert_called_once_with(event_data['regime'])


if __name__ == '__main__':
    unittest.main()