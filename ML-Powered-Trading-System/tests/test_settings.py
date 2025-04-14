"""
Test settings and constants.
"""

# Test environment settings
TEST_ENVIRONMENT = "test"
MOCK_EXCHANGES = True
USE_HISTORICAL_DATA = True

# Performance test thresholds
MAX_LATENCY_MS = {
    "market_data_processing": 10,
    "signal_generation": 50,
    "order_execution": 20,
    "critical_path": 100
}

MAX_MEMORY_USAGE_MB = 1024
MAX_CPU_USAGE_PERCENT = 80

# Test data paths
TEST_DATA_DIR = "tests/fixtures/data"
HISTORICAL_DATA_PATH = f"{TEST_DATA_DIR}/historical"
MARKET_DATA_SAMPLE_PATH = f"{TEST_DATA_DIR}/market_data_sample.csv"

# Mock response settings
MOCK_RESPONSE_DELAY_MS = 5
