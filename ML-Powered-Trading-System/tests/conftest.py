"""
Common pytest fixtures and configuration.
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch


@pytest.fixture(scope="session")
def test_environment():
    """Set up a test environment that's used across all tests."""
    # Set up environment
    env = {
        "test_mode": True,
        "paper_trading": True,
        "mocked_exchanges": True
    }

    yield env

    # Tear down if needed
    pass


@pytest.fixture
def market_data_sample():
    """Create sample market data for testing."""
    # Create a DataFrame with sample market data
    data = {
        'timestamp': pd.date_range(start='2023-01-01', periods=100, freq='1min'),
        'open': np.random.randn(100) * 10 + 100,
        'high': np.random.randn(100) * 10 + 105,
        'low': np.random.randn(100) * 10 + 95,
        'close': np.random.randn(100) * 10 + 101,
        'volume': np.random.randint(1000, 10000, 100)
    }
    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    return df


@pytest.fixture
def mock_event_bus():
    """Create a mock event bus for testing."""
    mock_bus = Mock()
    mock_bus.publish = Mock()
    mock_bus.subscribe = Mock()
    return mock_bus


@pytest.fixture
def mock_exchange_api():
    """Create a mock exchange API for testing."""
    mock_api = Mock()

    # Configure the mock to return realistic responses
    mock_api.get_order_book.return_value = {
        'bids': [(100.0, 1.5), (99.5, 2.3), (99.0, 3.2)],
        'asks': [(101.0, 1.2), (101.5, 2.1), (102.0, 3.5)]
    }

    mock_api.place_order.return_value = {
        'id': '12345',
        'status': 'open',
        'symbol': 'BTC-USD',
        'side': 'buy',
        'price': 100.0,
        'amount': 1.0
    }

    return mock_api
