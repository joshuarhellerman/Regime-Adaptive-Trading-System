import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import time
from datetime import datetime, timedelta

from models.strategies.momentum_strategy import MomentumStrategy
from models.strategies.strategy_base import DirectionalBias, VolatilityRegime
from core.event_bus import EventBus


@pytest.fixture
def sample_data():
    """Create a sample DataFrame with OHLC data for testing."""
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    
    # Generate price data with a trend
    close_prices = np.linspace(100, 120, 100) + np.sin(np.linspace(0, 10, 100)) * 5
    
    data = pd.DataFrame({
        'open': close_prices - np.random.rand(100),
        'high': close_prices + np.random.rand(100) * 2,
        'low': close_prices - np.random.rand(100) * 2,
        'close': close_prices,
        'volume': np.random.randint(1000, 10000, 100),
        'symbol': 'AAPL'
    }, index=dates)
    
    return data


@pytest.fixture
def strategy():
    """Create a MomentumStrategy instance for testing."""
    return MomentumStrategy(
        name="Test Momentum",
        parameters={
            'rsi_period': 14,
            'rsi_threshold': 55,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'roc_period': 10,
            'roc_threshold': 0.0,
            'volume_confirmation': True,
            'use_macd': True,
            'use_rsi': True,
            'use_roc': True,
            'min_momentum_score': 0.6,
            'min_bars': 50
        },
        strategy_id="test-momentum-strategy"
    )


@pytest.fixture
def mock_event_bus():
    """Mock the EventBus to track emitted events."""
    with patch('core.event_bus.EventBus') as mock_event_bus:
        yield mock_event_bus


class TestMomentumStrategy:
    
    def test_initialization(self, strategy):
        """Test that the strategy initializes correctly with expected parameters."""
        assert strategy.name == "Test Momentum"
        assert strategy.id == "test-momentum-strategy"
        assert strategy.parameters['rsi_period'] == 14
        assert strategy.parameters['macd_fast'] == 12
        assert strategy.parameters['macd_slow'] == 26
        assert strategy.parameters['roc_period'] == 10
        assert strategy.parameters['min_momentum_score'] == 0.6
        
        # Check that default state is set
        assert strategy._state['momentum_threshold'] == 0.6
        assert strategy._state['rsi_threshold_upper'] == 55
        assert strategy._state['rsi_threshold_lower'] == 45
        assert strategy._state['stop_loss_atr_multiplier'] == 2.0
        assert strategy._state['take_profit_atr_multiplier'] == 3.0
    
    def test_validate_parameters_valid(self, strategy):
        """Test parameter validation with valid parameters."""
        # This should not raise any errors
        strategy._validate_parameters()
    
    def test_validate_parameters_invalid(self):
        """Test parameter validation with invalid parameters."""
        # Test with invalid RSI threshold
        with pytest.raises(ValueError, match="RSI threshold must be between 50 and 70"):
            MomentumStrategy(parameters={'rsi_threshold': 40})
        
        # Test with invalid MACD periods
        with pytest.raises(ValueError, match="MACD fast period must be less than slow period"):
            MomentumStrategy(parameters={'macd_fast': 30, 'macd_slow': 26})
        
        # Test with invalid momentum score
        with pytest.raises(ValueError, match="Minimum momentum score must be between 0 and 1"):
            MomentumStrategy(parameters={'min_momentum_score': 1.5})
    
    def test_calculate_rsi(self, strategy, sample_data):
        """Test the RSI calculation method."""
        rsi = strategy._calculate_rsi(sample_data['close'])
        
        # Check that RSI has expected properties
        assert isinstance(rsi, pd.Series)
        assert len(rsi) == len(sample_data)
        assert all(0 <= val <= 100 for val in rsi.dropna())
    
    def test_calculate_macd(self, strategy, sample_data):
        """Test the MACD calculation method."""
        macd, signal, hist = strategy._calculate_macd(sample_data['close'])
        
        # Check that MACD components have expected properties
        assert isinstance(macd, pd.Series)
        assert isinstance(signal, pd.Series)
        assert isinstance(hist, pd.Series)
        assert len(macd) == len(sample_data)
        assert len(signal) == len(sample_data)
        assert len(hist) == len(sample_data)
        
        # Check that hist = macd - signal
        pd.testing.assert_series_equal(hist, macd - signal, check_names=False)
    
    def test_calculate_roc(self, strategy, sample_data):
        """Test the Rate of Change calculation method."""
        roc = strategy._calculate_roc(sample_data['close'])
        
        # Check that ROC has expected properties
        assert isinstance(roc, pd.Series)
        assert len(roc) == len(sample_data)
    
    def test_calculate_atr(self, strategy, sample_data):
        """Test the ATR calculation method."""
        atr = strategy._calculate_atr(sample_data)
        
        # Check that ATR has expected properties
        assert isinstance(atr, pd.Series)
        assert len(atr) == len(sample_data)
        assert all(val >= 0 for val in atr.dropna())
    
    def test_calculate_momentum_score(self, strategy, sample_data):
        """Test the momentum score calculation."""
        # First calculate indicators that are needed for momentum score
        indicators = {
            'rsi': strategy._calculate_rsi(sample_data['close']),
            'macd': strategy._calculate_macd(sample_data['close'])[0],
            'macd_signal': strategy._calculate_macd(sample_data['close'])[1],
            'macd_hist': strategy._calculate_macd(sample_data['close'])[2],
            'roc': strategy._calculate_roc(sample_data['close'])
        }
        
        # Calculate momentum score
        score, direction = strategy._calculate_momentum_score(sample_data, indicators)
        
        # Check that score is between 0 and 1
        assert 0 <= score <= 1
        # Check that direction is one of the expected values
        assert direction in ['up', 'down']
    
    def test_generate_signal_insufficient_data(self, strategy):
        """Test signal generation with insufficient data."""
        # Create minimal data with fewer bars than required
        short_data = pd.DataFrame({
            'open': [100] * 10,
            'high': [105] * 10,
            'low': [95] * 10,
            'close': [101] * 10,
            'volume': [1000] * 10
        })
        
        # Should return None due to insufficient data
        signal = strategy.generate_signal(short_data)
        assert signal is None
    
    @patch('time.time', return_value=1000.0)
    def test_generate_signal_long(self, mock_time, strategy, sample_data, mock_event_bus):
        """Test generation of a long signal."""
        # Modify sample data to create conditions for a long signal
        bullish_data = sample_data.copy()
        
        # Pre-calculate indicators to force a bullish scenario
        rsi = pd.Series([60.0] * len(bullish_data))
        macd = pd.Series([2.0] * len(bullish_data))
        macd_signal = pd.Series([1.0] * len(bullish_data))
        macd_hist = pd.Series([1.0] * len(bullish_data))
        roc = pd.Series([5.0] * len(bullish_data))
        
        bullish_data['rsi_14'] = rsi
        bullish_data['macd'] = macd
        bullish_data['macd_signal'] = macd_signal
        bullish_data['macd_hist'] = macd_hist
        bullish_data['roc_10'] = roc
        
        # Mock _calculate_momentum_score to return a high score
        with patch.object(strategy, '_calculate_momentum_score', return_value=(0.8, 'up')):
            with patch.object(strategy, '_is_optimal_entry_timing', return_value=True):
                with patch.object(strategy, '_confirm_volume', return_value=True):
                    # Generate signal
                    signal = strategy.generate_signal(bullish_data)
        
        # Should generate a long signal
        assert signal == 'long'
        
        # Check that event was emitted
        mock_event_bus.emit.assert_called_with("strategy.signal", {
            'strategy_id': strategy.id,
            'strategy_type': 'momentum',
            'timestamp': 1000.0,
            'signal': 'long',
            'instrument': 'AAPL',
            'price': bullish_data['close'].iloc[-1],
            'confidence': pytest.approx(0.6, abs=0.3),  # Allow some flexibility in confidence calculation
            'metadata': {
                'momentum_score': 0.8,
                'direction': 'up',
                'entry_optimization': 'default',
                'indicators': {
                    'rsi': 60.0,
                    'macd': 2.0,
                    'macd_signal': 1.0,
                    'macd_hist': 1.0,
                    'roc': 5.0
                }
            }
        })
    
    @patch('time.time', return_value=1000.0)
    def test_generate_signal_short(self, mock_time, strategy, sample_data, mock_event_bus):
        """Test generation of a short signal."""
        # Modify sample data to create conditions for a short signal
        bearish_data = sample_data.copy()
        
        # Pre-calculate indicators to force a bearish scenario
        rsi = pd.Series([40.0] * len(bearish_data))
        macd = pd.Series([-2.0] * len(bearish_data))
        macd_signal = pd.Series([-1.0] * len(bearish_data))
        macd_hist = pd.Series([-1.0] * len(bearish_data))
        roc = pd.Series([-5.0] * len(bearish_data))
        
        bearish_data['rsi_14'] = rsi
        bearish_data['macd'] = macd
        bearish_data['macd_signal'] = macd_signal
        bearish_data['macd_hist'] = macd_hist
        bearish_data['roc_10'] = roc
        
        # Mock _calculate_momentum_score to return a high score with downward direction
        with patch.object(strategy, '_calculate_momentum_score', return_value=(0.8, 'down')):
            with patch.object(strategy, '_is_optimal_entry_timing', return_value=True):
                with patch.object(strategy, '_confirm_volume', return_value=True):
                    # Generate signal
                    signal = strategy.generate_signal(bearish_data)
        
        # Should generate a short signal
        assert signal == 'short'
        
        # Check that event was emitted
        mock_event_bus.emit.assert_called_with("strategy.signal", {
            'strategy_id': strategy.id,
            'strategy_type': 'momentum',
            'timestamp': 1000.0,
            'signal': 'short',
            'instrument': 'AAPL',
            'price': bearish_data['close'].iloc[-1],
            'confidence': pytest.approx(0.6, abs=0.3),  # Allow some flexibility in confidence calculation
            'metadata': {
                'momentum_score': 0.8,
                'direction': 'down',
                'entry_optimization': 'default',
                'indicators': {
                    'rsi': 40.0,
                    'macd': -2.0,
                    'macd_signal': -1.0,
                    'macd_hist': -1.0,
                    'roc': -5.0
                }
            }
        })
    
    def test_generate_signal_no_signal(self, strategy, sample_data):
        """Test when no signal is generated due to low momentum score."""
        # Mock _calculate_momentum_score to return a low score
        with patch.object(strategy, '_calculate_momentum_score', return_value=(0.4, 'up')):
            # Generate signal
            signal = strategy.generate_signal(sample_data)
        
        # Should not generate a signal
        assert signal is None
    
    def test_optimal_entry_timing(self, strategy, sample_data):
        """Test the optimal entry timing function."""
        # Create indicators dict with different scenarios
        indicators = {
            'rsi': pd.Series([60.0, 65.0, 63.0]),
            'macd': pd.Series([1.0, 0.5, 1.5]),
            'macd_signal': pd.Series([0.8, 0.7, 0.9]),
            'macd_hist': pd.Series([0.2, -0.2, 0.6])
        }
        
        # Test case 1: MACD crossover (bullish)
        indicators['macd'].iloc[-2] = 0.5
        indicators['macd'].iloc[-1] = 1.0
        indicators['macd_signal'].iloc[-2] = 0.6
        indicators['macd_signal'].iloc[-1] = 0.5
        
        result = strategy._is_optimal_entry_timing(sample_data.iloc[-3:], indicators, 'up')
        assert result is True
        assert strategy._state['entry_optimization'] == 'breakout'
        
        # Test case 2: Pullback after uptrend
        test_data = sample_data.iloc[-3:].copy()
        test_data['close'].iloc[-3] = 100
        test_data['close'].iloc[-2] = 105
        test_data['close'].iloc[-1] = 104  # Small pullback
        
        result = strategy._is_optimal_entry_timing(test_data, indicators, 'up')
        assert result is True
        assert strategy._state['entry_optimization'] == 'pullback'
    
    def test_confirm_volume(self, strategy, sample_data):
        """Test the volume confirmation function."""
        # Create data with increasing volume trend
        data = sample_data.iloc[-10:].copy()
        data['volume'].iloc[-1] = data['volume'].iloc[-10:-1].mean() * 1.5  # Higher than average
        data['open'].iloc[-1] = 100
        data['close'].iloc[-1] = 105  # Up day
        
        result = strategy._confirm_volume(data, 'up')
        assert result is True
        
        # Test with declining volume on up day
        data['volume'].iloc[-1] = data['volume'].iloc[-10:-1].mean() * 0.5  # Lower than average
        result = strategy._confirm_volume(data, 'up')
        assert result is False
    
    @patch('time.time', return_value=1000.0)
    def test_risk_parameters(self, mock_time, strategy, sample_data, mock_event_bus):
        """Test the risk parameters calculation."""
        # Add ATR to the sample data
        sample_data['atr_14'] = 2.0
        
        # Set momentum strength in state
        strategy._state['current_momentum_score'] = 0.75
        
        # Calculate risk parameters
        entry_price = 110.0
        risk_params = strategy.risk_parameters(sample_data, entry_price)
        
        # Check that risk parameters have expected fields
        assert 'stop_loss_pct' in risk_params
        assert 'take_profit_pct' in risk_params
        assert 'position_size' in risk_params
        assert 'stop_price' in risk_params
        assert 'take_profit_price' in risk_params
        assert 'atr' in risk_params
        assert 'momentum_strength' in risk_params
        
        # Check that stop and target prices are calculated correctly
        assert risk_params['stop_price'] == pytest.approx(entry_price * (1 - risk_params['stop_loss_pct']))
        assert risk_params['take_profit_price'] == pytest.approx(entry_price * (1 + risk_params['take_profit_pct']))
        
        # Check that event was emitted
        mock_event_bus.emit.assert_called_with("strategy.risk_parameters", {
            'strategy_id': strategy.id,
            'strategy_type': 'momentum',
            'timestamp': 1000.0,
            'entry_price': entry_price,
            'stop_loss_pct': risk_params['stop_loss_pct'],
            'take_profit_pct': risk_params['take_profit_pct'],
            'position_size': risk_params['position_size'],
            'risk_reward_ratio': risk_params['take_profit_pct'] / risk_params['stop_loss_pct'],
            'momentum_strength': 0.75,
            'entry_optimization': 'default'
        })
    
    @patch('time.time', return_value=1000.0)
    def test_exit_signal_long(self, mock_time, strategy, sample_data, mock_event_bus):
        """Test exit signal for a long position."""
        # Create data with exit conditions for a long position
        data = sample_data.copy()
        data['rsi_14'] = pd.Series([40.0] * len(data))  # RSI below threshold
        data['macd'] = pd.Series([-1.0] * len(data))
        data['macd_signal'] = pd.Series([0.0] * len(data))
        data['macd_hist'] = pd.Series([-1.0] * len(data))  # Negative MACD histogram
        data['roc_10'] = pd.Series([-2.0] * len(data))  # Negative ROC
        
        # Create position dict
        position = {
            'id': 'position-123',
            'direction': 'long',
            'entry_price': 100.0,
            'entry_time': 900.0
        }
        
        # Test exit signal
        result = strategy.exit_signal(data, position)
        assert result is True
        
        # Check that exit event was emitted
        mock_event_bus.emit.assert_called_with("strategy.exit", {
            'strategy_id': strategy.id,
            'strategy_type': 'momentum',
            'timestamp': 1000.0,
            'position_id': 'position-123',
            'direction': 'long',
            'entry_price': 100.0,
            'current_price': data['close'].iloc[-1],
            'holding_time': 100.0,
            'reason': "Exit long: RSI declined below threshold (40.0)"
        })
    
    @patch('time.time', return_value=1000.0)
    def test_exit_signal_short(self, mock_time, strategy, sample_data, mock_event_bus):
        """Test exit signal for a short position."""
        # Create data with exit conditions for a short position
        data = sample_data.copy()
        data['rsi_14'] = pd.Series([60.0] * len(data))  # RSI above threshold
        data['macd'] = pd.Series([1.0] * len(data))
        data['macd_signal'] = pd.Series([0.0] * len(data))
        data['macd_hist'] = pd.Series([1.0] * len(data))  # Positive MACD histogram
        data['roc_10'] = pd.Series([2.0] * len(data))  # Positive ROC
        
        # Create position dict
        position = {
            'id': 'position-456',
            'direction': 'short',
            'entry_price': 120.0,
            'entry_time': 900.0
        }
        
        # Test exit signal
        result = strategy.exit_signal(data, position)
        assert result is True
        
        # Check that exit event was emitted
        mock_event_bus.emit.assert_called_with("strategy.exit", {
            'strategy_id': strategy.id,
            'strategy_type': 'momentum',
            'timestamp': 1000.0,
            'position_id': 'position-456',
            'direction': 'short',
            'entry_price': 120.0,
            'current_price': data['close'].iloc[-1],
            'holding_time': 100.0,
            'reason': "Exit short: RSI increased above threshold (60.0)"
        })
    
    @patch('time.time', return_value=1000.0)
    def test_on_trade_completed(self, mock_time, strategy, mock_event_bus):
        """Test the trade completion callback and parameter adaptation."""
        # Store original parameters
        original_momentum_threshold = strategy._state['momentum_threshold']
        original_stop_loss_multiplier = strategy._state['stop_loss_atr_multiplier']
        original_take_profit_multiplier = strategy._state['take_profit_atr_multiplier']
        
        # Create a successful trade result
        trade_result = {
            'id': 'trade-123',
            'pnl': 500.0,
            'pnl_pct': 0.06,  # 6% profit
            'regime_id': 'trend-01',
            'entry_optimization': 'breakout'
        }
        
        # Process trade
        strategy.on_trade_completed(trade_result)
        
        # Check that parameters were adjusted
        assert strategy._state['momentum_threshold'] != original_momentum_threshold
        assert strategy._state['take_profit_atr_multiplier'] != original_take_profit_multiplier
        
        # Check that event was emitted
        mock_event_bus.emit.assert_called_with("strategy.parameter_adaptation", {
            'strategy_id': strategy.id,
            'strategy_type': 'momentum',
            'timestamp': 1000.0,
            'trade_id': 'trade-123',
            'trade_pnl': 500.0,
            'trade_pnl_pct': 0.06,
            'regime_id': 'trend-01',
            'parameter_changes': {
                'momentum_threshold': {
                    'old': original_momentum_threshold,
                    'new': strategy._state['momentum_threshold']
                },
                'stop_loss_atr_multiplier': {
                    'old': original_stop_loss_multiplier,
                    'new': strategy._state['stop_loss_atr_multiplier']
                },
                'take_profit_atr_multiplier': {
                    'old': original_take_profit_multiplier,
                    'new': strategy._state['take_profit_atr_multiplier']
                },
                'entry_optimization': 'breakout'
            },
            'adaptation_method': 'online_bayesian'
        })
    
    def test_get_required_features(self, strategy):
        """Test that the strategy correctly reports required features."""
        features = strategy.get_required_features()
        assert isinstance(features, set)
        
        # Check that basic OHLCV features are required
        assert 'open' in features
        assert 'high' in features
        assert 'low' in features
        assert 'close' in features
        assert 'volume' in features
        
        # Check that indicator features are required
        assert 'rsi_14' in features
        assert 'macd' in features
        assert 'macd_signal' in features
        assert 'macd_hist' in features
        assert 'roc_10' in features
        assert 'atr_14' in features
    
    @patch('time.time', return_value=1000.0)
    def test_update_parameters_online(self, mock_time, strategy, mock_event_bus):
        """Test online parameter updates based on performance metrics."""
        # Store original parameters
        original_momentum_threshold = strategy._state['momentum_threshold']
        original_rsi_threshold_upper = strategy._state['rsi_threshold_upper']
        original_rsi_threshold_lower = strategy._state['rsi_threshold_lower']
        
        # Create performance metrics and market conditions
        performance_metrics = {
            'sharpe_ratio': 1.8,
            'win_rate': 0.6,
            'profit_factor': 1.6
        }
        
        market_conditions = {
            'adx': 30,
            'volatility': 0.6
        }
        
        # Update parameters
        strategy.update_parameters_online(performance_metrics, market_conditions)
        
        # Check that parameters were adjusted
        assert strategy._state['momentum_threshold'] != original_momentum_threshold
        assert strategy._state['rsi_threshold_upper'] != original_rsi_threshold_upper
        assert strategy._state['rsi_threshold_lower'] != original_rsi_threshold_lower
        
        # Check that event was emitted
        mock_event_bus.emit.assert_called_with("strategy.online_update", {
            'strategy_id': strategy.id,
            'strategy_type': 'momentum',
            'timestamp': 1000.0,
            'new_parameters': {
                'momentum_threshold': strategy._state['momentum_threshold'],
                'rsi_threshold_upper': strategy._state['rsi_threshold_upper'],
                'rsi_threshold_lower': strategy._state['rsi_threshold_lower'],
                'stop_loss_atr_multiplier': strategy._state['stop_loss_atr_multiplier'],
                'take_profit_atr_multiplier': strategy._state['take_profit_atr_multiplier']
            },
            'performance_metrics': performance_metrics,
            'market_conditions': market_conditions
        })
    
    @patch('time.time', return_value=1000.0)
    def test_adapt_to_regime(self, mock_time, strategy, mock_event_bus):
        """Test strategy adaptation to different market regimes."""
        # Store original parameters
        original_momentum_threshold = strategy._state['momentum_threshold']
        original_rsi_threshold_upper = strategy._state['rsi_threshold_upper']
        original_rsi_threshold_lower = strategy._state['rsi_threshold_lower']
        
        # Create regime data
        regime_data = {
            'id': 'high-vol-uptrend',
            'directional_bias': DirectionalBias.UPWARD,
            'volatility_regime': VolatilityRegime.HIGH,
            'peak_hour': 14
        }
        
        # Adapt to regime
        strategy.adapt_to_regime(regime_data)
        
        # Check that parameters were adjusted
        assert strategy._state['momentum_threshold'] != original_momentum_threshold
        assert strategy._state['rsi_threshold_upper'] != original_rsi_threshold_upper
        assert strategy._state['rsi_threshold_lower'] != original_rsi_threshold_lower
        
        # Check that regime adaptation event was emitted
        mock_event_bus.emit.assert_called_with("strategy.regime_adaptation", {
            'strategy_id': strategy.id,
            'strategy_type': 'momentum',
            'timestamp': 1000.0,
            'regime_id': 'high-vol-uptrend',
            'parameters': {
                'momentum_threshold': {
                    'old': original_momentum_threshold,
                    'new': strategy._state['momentum_threshold']
                },
                'rsi_threshold_upper': {
                    'old': original_rsi_threshold_upper,
                    'new': strategy._state['rsi_threshold_upper']
                },
                'rsi_threshold_lower': {
                    'old': original_rsi_threshold_lower,
                    'new': strategy._state['rsi_threshold_lower']
                },
                'stop_loss_atr_multiplier': strategy._state['stop_loss_atr_multiplier'],
                'take_profit_atr_multiplier': strategy._state['take_profit_atr_multiplier']
            }
        })
        
        # Also test with a different regime to ensure different adaptations
        strategy._state['momentum_threshold'] = original_momentum_threshold
        strategy._state['rsi_threshold_upper'] = original_rsi_threshold_upper
        strategy._state['rsi_threshold_lower'] = original_rsi_threshold_lower
        
        different_regime = {
            'id': 'low-vol-downtrend',
            'directional_bias': DirectionalBias.DOWNWARD,
            'volatility_regime': VolatilityRegime.LOW,
            'peak_hour': 9
        }
        
        strategy.adapt_to_regime(different_regime)
        
        # Parameters should be different for different regimes
        assert strategy._state['momentum_threshold'] != original_momentum_threshold
    
    @patch('time.time', return_value=1000.0)
    def test_cluster_fit(self, mock_time, strategy, mock_event_bus):
        """Test cluster fitness calculation."""
        # Create cluster metrics
        cluster_metrics = {
            'cluster_id': 'cluster-123',
            'ADX_mean': 25,
            'momentum_score': 0.7,
            'trend_persistence': 0.6,
            'volatility_pct_rank': 0.5
        }
        
        # Calculate fitness
        fitness = strategy.cluster_fit(cluster_metrics)
        
        # Fitness should be between 0 and 1
        assert 0 <= fitness <= 1
        
        # Check that event was emitted
        mock_event_bus.emit.assert_called_with("strategy.cluster_fit", {
            'strategy_id': strategy.id,
            'strategy_type': 'momentum',
            'cluster_id': 'cluster-123',
            'fitness_score': fitness,
            'components': {
                'adx_score': pytest.approx(0.9, abs=0.2),
                'momentum_score': pytest.approx(0.8, abs=0.2),
                'persistence_score': pytest.approx(0.8, abs=0.2),
                'volatility_score': pytest.approx(0.5, abs=0.2)
            }
        })


if __name__ == "__main__":
    pytest.main()