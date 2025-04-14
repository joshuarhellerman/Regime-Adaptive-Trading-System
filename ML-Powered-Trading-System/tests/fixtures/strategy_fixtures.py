"""
strategy_fixtures fixtures for testing.
"""
import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def sample_strategy():
    """
    Create a sample strategy for testing.
    """
    # Create and return test data
    return {
        "test_data": "sample_value"
    }


@pytest.fixture
def mock_strategy():
    """
    Create a mock strategy for testing.
    """
    # Create and return mock data
    return {
        "mock_data": "mock_value"
    }
