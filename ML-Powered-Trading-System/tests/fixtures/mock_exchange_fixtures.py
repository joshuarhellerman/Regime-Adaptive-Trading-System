"""
mock_exchange_fixtures fixtures for testing.
"""
import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def sample_mock_exchange():
    """
    Create a sample mock_exchange for testing.
    """
    # Create and return test data
    return {
        "test_data": "sample_value"
    }


@pytest.fixture
def mock_mock_exchange():
    """
    Create a mock mock_exchange for testing.
    """
    # Create and return mock data
    return {
        "mock_data": "mock_value"
    }
