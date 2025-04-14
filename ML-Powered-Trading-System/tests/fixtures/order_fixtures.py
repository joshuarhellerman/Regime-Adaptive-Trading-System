"""
order_fixtures fixtures for testing.
"""
import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def sample_order():
    """
    Create a sample order for testing.
    """
    # Create and return test data
    return {
        "test_data": "sample_value"
    }


@pytest.fixture
def mock_order():
    """
    Create a mock order for testing.
    """
    # Create and return mock data
    return {
        "mock_data": "mock_value"
    }
