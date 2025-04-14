"""
market_data_fixtures fixtures for testing.
"""
import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def sample_market_data():
    """
    Create a sample market_data for testing.
    """
    # Create and return test data
    return {
        "test_data": "sample_value"
    }


@pytest.fixture
def mock_market_data():
    """
    Create a mock market_data for testing.
    """
    # Create and return mock data
    return {
        "mock_data": "mock_value"
    }
