import unittest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
import time

from models.strategies.strategy_base import DirectionalBias, VolatilityRegime
from models.strategies.mean_reversion_strategy import MeanReversionStrategy

class TestMeanReversionStrategy(unittest.TestCase):
    """Unit tests for the MeanReversionStrategy class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Default strategy parameters
        self.default_params = {
            'lookback_period': 20,
            'rsi_period': 14,
            'bb_period': 20,
            'bb_std': 2.0,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'zscore_threshold': 2.0,
            'exit_zscore': 0.5,
            'min_liquidity': 1.0,
            'max_holding_period': 10,
            'volatility_filter': True,
            'min_bars': 30,
            'epsilon': 1e-8
        }
        
        # Create strategy instance with default parameters
        self.strategy = MeanReversionStrategy(name="Test Strategy", parameters=self.default_params)
        
        # Sample price data for testing
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        
        # Creating sample data with a price pattern that should trigger signals
        close_prices = np.linspace(100, 120, 50).tolist() + np.linspace(120, 90, 50).tolist()
        
        # Add some volatility
        np.random.seed(42)  # For reproducibility
        noise = np.random.normal(0, 2, 100)
        close_prices = [p + n for p, n in zip(close_prices, noise)]
        
        # Create basic OHLC data
        self.sample_data = pd.DataFrame({
            'open': close_prices,
            'high': [p * 1.02 for p in close_prices],
            'low': [p * 0.98 for p in close_prices],
            'close': close_prices,
            'volume': np.random.randint(1000, 10000, 100)
        }, index=dates)
        
    def test_initialization(self):
        """Test initialization with default and custom parameters."""
        # Test with default parameters
        strategy = MeanReversionStrategy()
        self.assertEqual(strategy.name, "MeanReversion")
        self.assertEqual(strategy.parameters['lookback_period'], 20)
        
        # Test with custom parameters
        custom_params = {'lookback_period': 30, 'rsi_period': 7}
        strategy = MeanReversionStrategy("Custom Strategy", custom_params)
        self.assertEqual(strategy.name, "Custom Strategy")
        self.assertEqual(strategy.parameters['lookback_period'], 30)
        self.assertEqual(strategy.parameters['rsi_period'], 7)
        # Default parameters should still be available
        self.assertEqual(strategy.parameters['bb_std'], 2.0)

    def test_parameter_validation(self):
        """Test parameter validation logic."""
        # Test invalid period value
        with self.assertRaises(ValueError):
            MeanReversionStrategy(parameters={'lookback_period': 0})
        
        # Test invalid RSI thresholds
        with self.assertRaises(ValueError):
            MeanReversionStrategy(parameters={'rsi_oversold': 50, 'rsi_overbought': 40})
        
        # Test invalid zscore threshold
        with self.assertRaises(ValueError):
            MeanReversionStrategy(parameters={'zscore_threshold': -1.0})
        
        # Test invalid BB std
        with self.assertRaises(ValueError):
            MeanReversionStrategy(parameters={'bb_std': 0})
            
        # Test invalid max holding period
        with self.assertRaises(ValueError):
            MeanReversionStrategy(parameters={'max_holding_period': 0})
            
        # Test invalid epsilon
        with self.assertRaises(ValueError):
            MeanReversionStrategy(parameters={'epsilon': 0})

    def test_data_validation(self):
        """Test data validation logic."""
        # Test empty data
        with self.assertRaises(ValueError):
            self.strategy.validate_data(pd.DataFrame())
        
        # Test missing required columns
        incomplete_data = pd.DataFrame({'close': [100, 101, 102]})
        with self.assertRaises(ValueError):
            self.strategy.validate_data(incomplete_data)
        
        # Test handling of NaN values
        data_with_nan = self.sample_data.copy()
        data_with_nan.loc[data_with_nan.index[5], 'close'] = np.nan
        
        # This should not raise an error as the validate_data method fills NaNs
        self.strategy.validate_data(data_with_nan)
        
        # Verify NaNs were filled
        self.assertFalse(data_with_nan['close'].isna().any())

    def test_indicator_calculation(self):
        """Test the indicator calculation functions."""
        # Test RSI calculation
        rsi = self.strategy._calculate_rsi(self.sample_data['close'])
        self.assertEqual(len(rsi), len(self.sample_data))
        self.assertTrue(all(0 <= val <= 100 for val in rsi.dropna()))
        
        # Test Bollinger Bands calculation
        upper, middle, lower = self.strategy._calculate_bollinger_bands(self.sample_data['close'])
        self.assertEqual(len(upper), len(self.sample_data))
        self.assertTrue(all(upper.dropna() >= middle.dropna()))
        self.assertTrue(all(lower.dropna() <= middle.dropna()))
        
        # Test ATR calculation
        atr = self.strategy._calculate_atr(self.sample_data)
        self.assertEqual(len(atr), len(self.sample_data))
        self.assertTrue(all(atr.dropna() >= 0))  # ATR should always be positive

    def test_cached_indicators(self):
        """Test the indicator caching mechanism."""
        # First call should calculate and cache
        rsi1 = self.strategy._get_cached_indicator('rsi', self.sample_data)
        
        # Second call should use cached value
        rsi2 = self.strategy._get_cached_indicator('rsi', self.sample_data)
        
        # Verify same object is returned
        self.assertIs(rsi1, rsi2)
        
        # Test cache for Bollinger Bands
        bb1 = self.strategy._get_cached_indicator('bollinger_bands', self.sample_data)
        bb2 = self.strategy._get_cached_indicator('bollinger_bands', self.sample_data)
        self.assertIs(bb1, bb2)
        
        # Test clearing cache
        self.strategy.clear_cache()
        self.assertEqual(len(self.strategy.indicator_cache), 0)
        
        # Test invalid indicator name
        with self.assertRaises(ValueError):
            self.strategy._get_cached_indicator('invalid_indicator', self.sample_data)

    def test_generate_signal_with_precomputed_indicators(self):
        """Test signal generation with precomputed indicators."""
        # Prepare data with precomputed indicators
        data = self.sample_data.copy()
        
        # Add RSI values
        data['rsi_14'] = self.strategy._calculate_rsi(data['close'], 14)
        
        # Add Bollinger Bands
        upper, middle, lower = self.strategy._calculate_bollinger_bands(data['close'], 20, 2.0)
        data['bb_upper'] = upper
        data['bb_lower'] = lower
        
        # Create conditions for a long signal (oversold)
        data.loc[data.index[50], 'rsi_14'] = 25  # Very oversold
        data.loc[data.index[50], 'close'] = lower[50] * 0.95  # Below lower BB
        
        # Generate signal
        signal = self.strategy.generate_signal(data.iloc[:51])
        self.assertEqual(signal, 'long')
        
        # Create conditions for a short signal (overbought)
        data.loc[data.index[60], 'rsi_14'] = 75  # Very overbought
        data.loc[data.index[60], 'close'] = upper[60] * 1.05  # Above upper BB
        
        # Generate signal
        signal = self.strategy.generate_signal(data.iloc[:61])
        self.assertEqual(signal, 'short')
        
        # Test no signal condition
        data.loc[data.index[70], 'rsi_14'] = 50  # Neutral RSI
        data.loc[data.index[70], 'close'] = middle[70]  # At middle BB
        
        # Generate signal - should be None
        signal = self.strategy.generate_signal(data.iloc[:71])
        self.assertIsNone(signal)

    def test_generate_signal_with_calculated_indicators(self):
        """Test signal generation with indicators calculated on the fly."""
        # Create conditions for signals using mock indicators
        with patch.object(self.strategy, '_get_cached_indicator') as mock_indicator:
            # Set up mock returns for indicators
            
            # Mock RSI - first return oversold, then overbought
            mock_rsi = pd.Series([25, 75], index=self.sample_data.index[[50, 60]])
            
            # Mock Bollinger Bands
            mock_upper = pd.Series([120, 120], index=self.sample_data.index[[50, 60]])
            mock_middle = pd.Series([110, 110], index=self.sample_data.index[[50, 60]])
            mock_lower = pd.Series([100, 100], index=self.sample_data.index[[50, 60]])
            
            # Configure mock to return different values based on indicator name
            def side_effect(indicator_name, *args):
                if indicator_name == 'rsi':
                    return mock_rsi
                elif indicator_name == 'bollinger_bands':
                    return (mock_upper, mock_middle, mock_lower)
                elif indicator_name == 'atr':
                    return pd.Series([2.0, 2.0], index=self.sample_data.index[[50, 60]])
            
            mock_indicator.side_effect = side_effect
            
            # Create test data
            data = self.sample_data.copy()
            
            # Test long signal
            data.loc[data.index[50], 'close'] = 95  # Below lower BB and oversold
            signal = self.strategy.generate_signal(data.iloc[:51])
            self.assertEqual(signal, 'long')
            
            # Test short signal
            data.loc[data.index[60], 'close'] = 125  # Above upper BB and overbought
            signal = self.strategy.generate_signal(data.iloc[:61])
            self.assertEqual(signal, 'short')

    def test_risk_parameters(self):
        """Test risk parameter calculation."""
        # Prepare data
        data = self.sample_data.copy()
        
        # Add ATR
        data['atr_14'] = self.strategy._calculate_atr(data, 14)
        
        # Test long position (below mean)
        entry_price = 95
        mean_price = data['close'].rolling(20).mean().iloc[-1]
        self.assertTrue(entry_price < mean_price)  # Confirm test assumption
        
        risk_params = self.strategy.risk_parameters(data, entry_price)
        
        # Verify risk parameters
        self.assertIn('stop_loss_pct', risk_params)
        self.assertIn('take_profit_pct', risk_params)
        self.assertIn('position_size', risk_params)
        
        # Stop loss should be positive
        self.assertGreater(risk_params['stop_loss_pct'], 0)
        
        # Take profit should be positive and reasonable (not too large)
        self.assertGreater(risk_params['take_profit_pct'], 0)
        self.assertLess(risk_params['take_profit_pct'], 0.2)  # 20% max
        
        # Test short position (above mean)
        entry_price = 120
        self.assertTrue(entry_price > mean_price)  # Confirm test assumption
        
        risk_params = self.strategy.risk_parameters(data, entry_price)
        
        # Verify risk parameters for short position
        self.assertGreater(risk_params['stop_loss_pct'], 0)
        self.assertGreater(risk_params['take_profit_pct'], 0)
        self.assertLess(risk_params['take_profit_pct'], 0.2)

    def test_exit_signal(self):
        """Test exit signal generation."""
        # Create data
        data = self.sample_data.copy()
        
        # Add Z-score directly for testing
        lookback = self.strategy.parameters['lookback_period']
        rolling_mean = data['close'].rolling(lookback).mean()
        rolling_std = data['close'].rolling(lookback).std() + self.strategy.parameters['epsilon']
        data['z_score'] = (data['close'] - rolling_mean) / rolling_std
        
        # Create position for long trade
        position = {
            'direction': 'long',
            'entry_price': 100,
            'entry_time': time.time() - 3600  # 1 hour ago
        }
        
        # Test exit triggered by Z-score for long position
        data.loc[data.index[-1], 'z_score'] = 0.6  # Above exit threshold
        self.assertTrue(self.strategy.exit_signal(data, position))
        
        # Create position for short trade
        position = {
            'direction': 'short',
            'entry_price': 100,
            'entry_time': time.time() - 3600  # 1 hour ago
        }
        
        # Test exit triggered by Z-score for short position
        data.loc[data.index[-1], 'z_score'] = -0.6  # Below negative exit threshold
        self.assertTrue(self.strategy.exit_signal(data, position))
        
        # Test no exit
        data.loc[data.index[-1], 'z_score'] = 0.1  # Between exit thresholds
        self.assertFalse(self.strategy.exit_signal(data, position))
        
        # Test exit due to max holding period
        old_entry_time = time.time() - (self.strategy.parameters['max_holding_period'] + 1) * 24 * 3600
        position['entry_time'] = old_entry_time
        self.assertTrue(self.strategy.exit_signal(data, position))

    def test_regime_adaptations(self):
        """Test adaptations to different market regimes."""
        # Setup mock regime characteristics
        self.strategy.regime_characteristics = MagicMock()
        
        # Test time of day optimization
        for peak_hour in [8, 9, 11]:
            self.strategy.regime_characteristics.peak_hour = peak_hour
            self.strategy._optimize_for_time_of_day()
            
            if peak_hour in [8, 9]:
                self.assertEqual(self.strategy.adaptive_thresholds, (35, 65))
                self.assertEqual(self.strategy.liquidity_factor, 0.8)
            elif peak_hour == 11:
                self.assertEqual(self.strategy.adaptive_thresholds, (25, 75))
                self.assertEqual(self.strategy.liquidity_factor, 1.2)
        
        # Test directional bias adaptation
        # Reset thresholds
        self.strategy.adaptive_thresholds = (30, 70)
        
        # Test upward bias
        self.strategy.regime_characteristics.directional_bias = DirectionalBias.UPWARD
        self.strategy._optimize_for_directional_bias()
        self.assertEqual(self.strategy.adaptive_thresholds, (25, 70))  # Lower oversold
        
        # Reset thresholds
        self.strategy.adaptive_thresholds = (30, 70)
        
        # Test downward bias
        self.strategy.regime_characteristics.directional_bias = DirectionalBias.DOWNWARD
        self.strategy._optimize_for_directional_bias()
        self.assertEqual(self.strategy.adaptive_thresholds, (30, 65))  # Lower overbought
        
        # Test volatility regime adaptation
        # Reset thresholds
        self.strategy.adaptive_thresholds = (30, 70)
        
        # Test low volatility
        self.strategy.regime_characteristics.volatility_regime = VolatilityRegime.LOW
        self.strategy._optimize_for_volatility_regime()
        self.assertEqual(self.strategy.adaptive_thresholds, (35, 65))  # Tighter thresholds
        
        # Reset thresholds
        self.strategy.adaptive_thresholds = (30, 70)
        
        # Test high volatility
        self.strategy.regime_characteristics.volatility_regime = VolatilityRegime.HIGH
        self.strategy._optimize_for_volatility_regime()
        self.assertEqual(self.strategy.adaptive_thresholds, (25, 75))  # Wider thresholds

    def test_on_trade_completed(self):
        """Test adaptation after completed trades."""
        # Setup initial thresholds
        self.strategy.adaptive_thresholds = (30, 70)
        self.strategy.parameters['exit_zscore'] = 0.5
        self.strategy.regime_characteristics = MagicMock()
        self.strategy.regime_characteristics.cluster_id = "test_cluster"
        
        # Test successful long trade
        trade_result = {
            'pnl': 100,
            'pnl_pct': 0.06,
            'direction': 'long',
            'regime_id': "test_cluster"
        }
        
        self.strategy.on_trade_completed(trade_result)
        self.assertEqual(self.strategy.adaptive_thresholds, (31, 70))  # Adjusted oversold threshold
        self.assertAlmostEqual(self.strategy.parameters['exit_zscore'], 0.3, delta=0.01)  # Adjusted exit threshold
        
        # Reset parameters
        self.strategy.adaptive_thresholds = (30, 70)
        self.strategy.parameters['exit_zscore'] = 0.5
        
        # Test unsuccessful short trade
        trade_result = {
            'pnl': -50,
            'pnl_pct': -0.04,
            'direction': 'short',
            'regime_id': "test_cluster"
        }
        
        self.strategy.on_trade_completed(trade_result)
        self.assertEqual(self.strategy.adaptive_thresholds, (30, 71))  # Adjusted overbought threshold
        self.assertAlmostEqual(self.strategy.parameters['exit_zscore'], 0.525, delta=0.01)  # Adjusted exit threshold

    def test_cluster_fit(self):
        """Test cluster fit determination."""
        # Test with low ADX (suitable for mean reversion)
        cluster_metrics = {'ADX_mean': 20}
        self.assertTrue(self.strategy.cluster_fit(cluster_metrics))
        
        # Test with negative autocorrelation (suitable for mean reversion)
        cluster_metrics = {'autocorrelation': -0.2}
        self.assertTrue(self.strategy.cluster_fit(cluster_metrics))
        
        # Test with high volatility (suitable for mean reversion)
        cluster_metrics = {'volatility_rank': 0.5}
        self.assertTrue(self.strategy.cluster_fit(cluster_metrics))
        
        # Test with high ADX (less suitable, but should still return True as fallback)
        cluster_metrics = {'ADX_mean': 30, 'autocorrelation': 0.1, 'volatility_rank': 0.2}
        self.assertTrue(self.strategy.cluster_fit(cluster_metrics))
        
        # Test with empty metrics (should return True as fallback)
        self.assertTrue(self.strategy.cluster_fit({}))
        
        # Test with None (should return True as fallback)
        self.assertTrue(self.strategy.cluster_fit(None))

    def test_get_performance_metrics(self):
        """Test performance metrics calculation."""
        # Basic metrics only
        metrics = self.strategy.get_performance_metrics()
        self.assertEqual(metrics['name'], "Test Strategy")
        self.assertEqual(metrics['adaptive_thresholds'], self.strategy.adaptive_thresholds)
        self.assertEqual(metrics['exit_zscore'], self.strategy.parameters['exit_zscore'])
        
        # Add trade history and test advanced metrics
        self.strategy.trade_history = [
            {'pnl': 100, 'pnl_pct': 0.05},
            {'pnl': 150, 'pnl_pct': 0.07},
            {'pnl': -80, 'pnl_pct': -0.04}
        ]
        
        metrics = self.strategy.get_performance_metrics()
        self.assertEqual(metrics['total_trades'], 3)
        self.assertAlmostEqual(metrics['win_rate'], 2/3)
        self.assertAlmostEqual(metrics['avg_profit'], 0.06)  # (0.05 + 0.07) / 2
        self.assertAlmostEqual(metrics['avg_loss'], -0.04)  # Only one loss
        
        # Test expectancy calculation
        expected_expectancy = (2/3 * 0.06) + (1/3 * -0.04)
        self.assertAlmostEqual(metrics['expectancy'], expected_expectancy)
        
        # Test profit factor
        expected_profit_factor = -(0.05 + 0.07) / -0.04
        self.assertAlmostEqual(metrics['profit_factor'], expected_profit_factor)

if __name__ == '__main__':
    unittest.main()