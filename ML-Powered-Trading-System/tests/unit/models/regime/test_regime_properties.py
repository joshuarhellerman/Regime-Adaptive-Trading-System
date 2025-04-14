import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from models.regime.regime_properties import RegimePropertiesCalculator


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    # Create a datetime index
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    
    # Create price data with different characteristics for different regimes
    np.random.seed(42)  # For reproducibility
    
    # Base price
    close = 100 + np.cumsum(np.random.normal(0, 1, 100))
    
    # Create dataframe with OHLCV data
    df = pd.DataFrame({
        'open': close[:-1].tolist() + [close[-1]],  # Shift by 1
        'high': close + np.random.uniform(0.5, 2, 100),
        'low': close - np.random.uniform(0.5, 2, 100),
        'close': close,
        'volume': np.random.randint(1000, 10000, 100),
        # Add some features
        'feature1': np.random.normal(0, 1, 100),
        'feature2': np.random.uniform(-1, 1, 100),
        'returns': np.random.normal(0.001, 0.01, 100),  # Add returns directly
    }, index=dates)
    
    # Add hour and day_of_week
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    
    return df


def test_init():
    """Test initialization of RegimePropertiesCalculator."""
    calculator = RegimePropertiesCalculator()
    assert calculator.memory_decay == 0.95
    assert calculator.min_samples == 50
    
    # Custom params
    calculator = RegimePropertiesCalculator(memory_decay=0.9, min_samples=30)
    assert calculator.memory_decay == 0.9
    assert calculator.min_samples == 30


def test_prepare_data(sample_data):
    """Test data preparation functionality."""
    # Create calculator
    calculator = RegimePropertiesCalculator()
    
    # Test with minimal data
    minimal_df = pd.DataFrame({'price': [100, 101, 102]})
    prepared_df = calculator._prepare_data(minimal_df)
    
    # Check that close, returns and volatility were added
    assert 'close' in prepared_df.columns
    assert 'returns' in prepared_df.columns
    assert 'volatility' in prepared_df.columns
    
    # Test with complete data
    prepared_sample = calculator._prepare_data(sample_data)
    # Should not modify original data
    assert id(prepared_sample) != id(sample_data)
    # Should already have all required columns
    assert 'returns' in prepared_sample.columns
    assert 'hour' in prepared_sample.columns
    assert 'day_of_week' in prepared_sample.columns


def test_update_with_insufficient_samples():
    """Test update with insufficient samples."""
    calculator = RegimePropertiesCalculator()
    
    # Create minimal dataframe
    df = pd.DataFrame({
        'close': [100, 101, 102],
        'returns': [0.01, 0.01, 0.01]
    })
    
    # First update should just track samples and timestamp
    result = calculator.update(df, regime_label=0)
    
    assert 'sample_count' in result
    assert result['sample_count'] == 1
    assert 'last_seen' in result
    
    # Properties shouldn't be calculated yet
    assert 'mean_return' not in result


def test_update_with_sufficient_samples(sample_data):
    """Test update with sufficient samples."""
    calculator = RegimePropertiesCalculator(min_samples=10)  # Lower threshold for testing
    
    # Need to artificially boost sample count to trigger full calculation
    calculator.regime_samples[0] = 10
    
    # Update with regime 0
    result = calculator.update(sample_data, regime_label=0)
    
    # Check that properties were calculated
    assert 'mean_return' in result
    assert 'return_volatility' in result
    assert 'volatility_level' in result
    assert 'sample_count' in result
    assert 'confidence' in result
    
    # Check history was updated
    assert 0 in calculator.regime_properties_history
    assert len(calculator.regime_properties_history[0]) == 1


def test_return_properties_calculation(sample_data):
    """Test calculation of return properties."""
    calculator = RegimePropertiesCalculator()
    
    # Extract method for testing
    properties = calculator._calculate_return_properties(sample_data, regime_label=0)
    
    # Check essential properties
    assert 'mean_return' in properties
    assert 'return_volatility' in properties
    assert 'sharpe_ratio' in properties
    assert 'win_rate' in properties
    assert 'profit_factor' in properties
    assert 'max_drawdown' in properties
    
    # Basic sanity checks
    assert -1 <= properties['sharpe_ratio'] <= 1  # Example constraint
    assert 0 <= properties['win_rate'] <= 1
    assert properties['profit_factor'] >= 0
    assert properties['max_drawdown'] <= 0


def test_volatility_properties_calculation(sample_data):
    """Test calculation of volatility properties."""
    calculator = RegimePropertiesCalculator()
    
    # Extract method for testing
    properties = calculator._calculate_volatility_properties(sample_data, regime_label=0)
    
    # Check volatility properties
    assert 'volatility_level' in properties
    assert 'volatility_of_volatility' in properties
    assert 'garch_persistence' in properties
    assert 'range_intensity' in properties
    
    # Basic sanity checks
    assert properties['volatility_level'] >= 0
    assert properties['volatility_of_volatility'] >= 0
    assert -1 <= properties['garch_persistence'] <= 1
    assert properties['range_intensity'] > 0


def test_correlation_properties_calculation(sample_data):
    """Test calculation of correlation properties."""
    calculator = RegimePropertiesCalculator()
    
    # Extract method for testing
    properties = calculator._calculate_correlation_properties(sample_data, regime_label=0)
    
    # Check correlation properties
    assert 'internal_correlation' in properties
    
    # Basic sanity checks
    assert -1 <= properties['internal_correlation'] <= 1


def test_temporal_properties_calculation(sample_data):
    """Test calculation of temporal properties."""
    calculator = RegimePropertiesCalculator()
    
    # Extract method for testing
    timestamp = pd.Timestamp('2023-03-01')
    properties = calculator._calculate_temporal_properties(sample_data, regime_label=0, timestamp=timestamp)
    
    # Check when history exists
    calculator.regime_properties_history[0].append({
        'timestamp': pd.Timestamp('2023-02-01'),
        'properties': {'some': 'property'}
    })
    properties = calculator._calculate_temporal_properties(sample_data, regime_label=0, timestamp=timestamp)
    
    # Check time-based properties
    assert 'peak_hour' in properties
    assert 'time_of_day_bias' in properties
    assert 'peak_day' in properties
    assert 'day_of_week_bias' in properties
    
    # Basic sanity checks
    assert 0 <= properties['peak_hour'] <= 23
    assert 0 <= properties['time_of_day_bias'] <= 1
    assert 0 <= properties['peak_day'] <= 6
    assert 0 <= properties['day_of_week_bias'] <= 1


def test_update_feature_stats(sample_data):
    """Test updating feature statistics."""
    calculator = RegimePropertiesCalculator()
    
    # Update feature stats
    calculator._update_feature_stats(sample_data, regime_label=0)
    
    # Check feature stats
    assert 0 in calculator.feature_stats
    assert 'feature1' in calculator.feature_stats[0]
    assert 'feature2' in calculator.feature_stats[0]
    
    # Check stats content
    feature_stats = calculator.feature_stats[0]['feature1']
    assert 'mean' in feature_stats
    assert 'std' in feature_stats
    assert 'skew' in feature_stats
    assert 'kurtosis' in feature_stats
    assert 'percentiles' in feature_stats


def test_record_transition():
    """Test recording regime transitions."""
    calculator = RegimePropertiesCalculator()
    
    # Record a transition
    timestamp = pd.Timestamp('2023-03-01')
    calculator.record_transition(from_regime=0, to_regime=1, timestamp=timestamp)
    
    # Check transition was recorded
    assert len(calculator.transitions) == 1
    assert calculator.transitions[0]['from_regime'] == 0
    assert calculator.transitions[0]['to_regime'] == 1
    assert calculator.transitions[0]['timestamp'] == timestamp
    
    # Add more transitions and check matrix update
    calculator.record_transition(from_regime=1, to_regime=2)
    calculator.record_transition(from_regime=2, to_regime=0)
    
    # Matrix should be created
    assert calculator.transition_matrix is not None
    assert 'regimes' in calculator.transition_matrix
    assert 'matrix' in calculator.transition_matrix
    
    # Check matrix dimensions
    matrix = calculator.transition_matrix['matrix']
    assert matrix.shape == (3, 3)  # 3x3 for regimes 0,1,2
    
    # Check probabilities sum to 1 for each row
    assert np.allclose(matrix.sum(axis=1), 1.0)


def test_get_feature_distribution():
    """Test getting feature distribution."""
    calculator = RegimePropertiesCalculator()
    
    # Add some feature stats
    calculator.feature_stats[0]['feature1'] = {
        'mean': 0.5,
        'std': 0.1,
        'percentiles': {'50': 0.5}
    }
    
    # Get distribution
    result = calculator.get_feature_distribution(regime_label=0, feature='feature1')
    
    # Check result
    assert result['mean'] == 0.5
    assert result['std'] == 0.1
    
    # Non-existent feature should return empty dict
    assert calculator.get_feature_distribution(regime_label=0, feature='nonexistent') == {}
    assert calculator.get_feature_distribution(regime_label=999, feature='feature1') == {}


def test_get_transition_probability():
    """Test getting transition probabilities."""
    calculator = RegimePropertiesCalculator()
    
    # Without a transition matrix, should return 0
    assert calculator.get_transition_probability(from_regime=0, to_regime=1) == 0
    
    # Add transitions to create matrix
    calculator.record_transition(from_regime=0, to_regime=1)
    calculator.record_transition(from_regime=0, to_regime=1)
    calculator.record_transition(from_regime=0, to_regime=2)
    calculator.record_transition(from_regime=1, to_regime=0)
    
    # Matrix should be created now
    assert calculator.transition_matrix is not None


def test_memory_decay():
    """Test memory decay in property updates."""
    calculator = RegimePropertiesCalculator(memory_decay=0.8)
    
    # Create dataframe for regime 0
    df1 = pd.DataFrame({
        'returns': [0.01] * 20,  # Constant positive returns
        'close': np.linspace(100, 120, 20)
    })
    
    # Create dataframe for second update with different returns
    df2 = pd.DataFrame({
        'returns': [-0.01] * 20,  # Constant negative returns
        'close': np.linspace(120, 100, 20)
    })
    
    # Set sample count high enough to trigger calculations
    calculator.regime_samples[0] = 20
    
    # First update
    result1 = calculator.update(df1, regime_label=0)
    mean_return1 = result1['mean_return']
    
    # Second update
    result2 = calculator.update(df2, regime_label=0)
    mean_return2 = result2['mean_return']
    
    # Mean return should have decayed toward the new value
    assert mean_return1 > 0  # First update should have positive mean
    assert mean_return2 < mean_return1  # Second should be lower due to negative returns
    
    # However, memory decay should prevent it from immediately becoming negative
    assert mean_return2 > -0.01  # Not fully negative yet due to decay


def test_regime_property_history():
    """Test storing and managing regime property history."""
    calculator = RegimePropertiesCalculator(property_history_size=3)
    
    # Create sample dataframe
    df = pd.DataFrame({
        'returns': [0.01] * 20,
        'close': np.linspace(100, 120, 20)
    })
    
    # Set sample count high enough
    calculator.regime_samples[0] = 20
    
    # Multiple updates with different timestamps
    for i in range(5):
        timestamp = pd.Timestamp(f"2023-01-{i+1}")
        calculator.update(df, regime_label=0, timestamp=timestamp)
        
    # Check history length is capped at property_history_size
    assert len(calculator.regime_properties_history[0]) == 3
    
    # Check timestamps are in ascending order (most recent at the end)
    timestamps = [h['timestamp'] for h in calculator.regime_properties_history[0]]
    assert timestamps[0] < timestamps[1] < timestamps[2]
    
    # Check most recent is 2023-01-05
    assert timestamps[2] == pd.Timestamp("2023-01-5")


if __name__ == "__main__":
    pytest.main(["-v", "test_regime_properties.py"])