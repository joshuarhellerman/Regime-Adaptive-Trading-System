#!/usr/bin/env python3
"""
Script to set up a comprehensive test directory structure for an ML-powered trading system.
This creates the folder structure and template test files based on the system architecture.
"""

import os
import shutil
from pathlib import Path


def create_file(filepath, content):
    """Create a file with the given content."""
    with open(filepath, 'w') as f:
        f.write(content)
    print(f"Created: {filepath}")


def create_test_file(filepath, module_name, class_name=None):
    """Create a test file with a basic template."""
    if class_name:
        content = f'''"""
Tests for the {module_name} module.
"""
import pytest
from unittest.mock import Mock, patch

# Import the module to test
# from core.{module_name} import {class_name}


class Test{class_name}:
    """Tests for the {class_name} class."""

    def setup_method(self):
        """Set up test fixtures."""
        # self.{module_name.lower()} = {class_name}()
        pass

    def test_initialization(self):
        """Test that the {class_name} initializes correctly."""
        # assert self.{module_name.lower()} is not None
        pass

    def test_basic_functionality(self):
        """Test the basic functionality of {class_name}."""
        # Test the main functionality
        pass

    def test_error_handling(self):
        """Test how {class_name} handles errors."""
        # Test error cases
        pass
'''
    else:
        content = f'''"""
Tests for the {module_name} module.
"""
import pytest
from unittest.mock import Mock, patch

# Import the module to test
# from {module_name.split("/")[-1]} import some_function


def test_basic_functionality():
    """Test the basic functionality of {module_name}."""
    # Test the main functionality
    assert True


def test_error_handling():
    """Test error handling in {module_name}."""
    # Test error cases
    assert True
'''
    create_file(filepath, content)


def create_fixture_file(filepath, fixture_name):
    """Create a fixture file with a basic template."""
    content = f'''"""
{fixture_name} fixtures for testing.
"""
import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def sample_{fixture_name.lower().replace("_fixtures", "")}():
    """
    Create a sample {fixture_name.lower().replace("_fixtures", "")} for testing.
    """
    # Create and return test data
    return {{
        "test_data": "sample_value"
    }}


@pytest.fixture
def mock_{fixture_name.lower().replace("_fixtures", "")}():
    """
    Create a mock {fixture_name.lower().replace("_fixtures", "")} for testing.
    """
    # Create and return mock data
    return {{
        "mock_data": "mock_value"
    }}
'''
    create_file(filepath, content)


def create_utility_file(filepath, utility_name):
    """Create a utility file with a basic template."""
    content = f'''"""
{utility_name} utilities for testing.
"""

def {utility_name.lower().replace("_", "")}(input_data):
    """
    Utility function for {utility_name.lower().replace("_", "")}.

    Args:
        input_data: Input data to process

    Returns:
        Processed data
    """
    # Implement utility function
    return input_data
'''
    create_file(filepath, content)


def create_conftest_file(filepath):
    """Create a conftest.py file with common fixtures."""
    content = '''"""
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
'''
    create_file(filepath, content)


def setup_test_structure():
    """Create the entire test directory structure."""
    root_dir = Path("ML-Powered-Trading-System/tests")

    # Delete existing tests directory if it exists
    if root_dir.exists():
        shutil.rmtree(root_dir)

    # Create main test directories
    unit_dir = root_dir / "unit"
    integration_dir = root_dir / "integration"
    performance_dir = root_dir / "performance"
    fixtures_dir = root_dir / "fixtures"
    utils_dir = root_dir / "utils"

    # Create common files
    os.makedirs(root_dir)
    create_conftest_file(root_dir / "conftest.py")
    create_file(root_dir / "requirements-test.txt",
                "pytest==7.3.1\npytest-cov==4.1.0\npytest-mock==3.10.0\nnumpy==1.24.3\npandas==2.0.2\n")

    run_tests_content = '''#!/usr/bin/env python3
"""
Test runner script for ML-powered trading system.
"""
import argparse
import os
import sys
import subprocess


def run_tests(category=None, module=None, verbose=False, coverage=False):
    """
    Run tests with the specified configuration.

    Args:
        category: Test category (unit, integration, performance)
        module: Specific module to test
        verbose: Whether to show verbose output
        coverage: Whether to generate coverage report
    """
    cmd = ["pytest"]

    if verbose:
        cmd.append("-v")

    if coverage:
        cmd.append("--cov=.")
        cmd.append("--cov-report=term")
        cmd.append("--cov-report=html")

    if category:
        if module:
            cmd.append(f"tests/{category}/{module}")
        else:
            cmd.append(f"tests/{category}")
    elif module:
        cmd.append(f"tests/unit/{module}")
        cmd.append(f"tests/integration/*{module}*")

    print(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run tests for ML-powered trading system")
    parser.add_argument("--category", choices=["unit", "integration", "performance"],
                        help="Test category to run")
    parser.add_argument("--module", help="Specific module to test (e.g., core, data)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show verbose output")
    parser.add_argument("--coverage", action="store_true", help="Generate coverage report")

    args = parser.parse_args()
    run_tests(args.category, args.module, args.verbose, args.coverage)
'''
    create_file(root_dir / "run_tests.py", run_tests_content)
    os.chmod(root_dir / "run_tests.py", 0o755)  # Make executable

    test_settings_content = '''"""
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
'''
    create_file(root_dir / "test_settings.py", test_settings_content)

    # Create unit test directory structure
    unit_modules = {
        "core": ["system", "event_bus", "state_manager", "risk_manager", "performance_metrics",
                 "scheduler", "trading_mode_controller", "component_registry", "disaster_recovery",
                 "health_monitor", "circuit_breaker"],

        "data": ["market_data_service", "exchange_connector", "historical_repository",
                 "alternative_data_gateway", "mock_data_provider", "data_normalizer",
                 "tick_aggregator", "feature_engineering", "feature_store", "time_series_store",
                 "market_snapshot", "persistent_queue", "data_integrity", "replay_service"],

        "models": {
            "": ["alpha_model_interface", "alpha_factory", "alpha_registry", "alpha_result",
                 "signal_combiner", "strategy_protocol", "strategy_context"],
            "strategies": ["breakout_strategy", "momentum_strategy", "mean_reversion_strategy",
                           "trend_following_strategy", "volatility_strategy"],
            "regime": ["classifier", "transition_detector", "properties", "adaptor"],
            "portfolio": ["risk_model", "optimizer", "allocation_manager", "trade_generator"],
            "research": ["research_environment", "hyperparameter_optimization", "feature_analyzer",
                         "model_validator", "cross_validation"],
            "production": ["model_deployer", "prediction_service", "model_monitoring"]
        },

        "execution": {
            "": ["execution_service"],
            "order": ["order", "order_factory", "order_book", "execution_algorithm", "twap_algorithm",
                      "vwap_algorithm", "smart_router"],
            "exchange": {
                "": ["exchange_gateway", "exchange_simulator", "rate_limiter", "connectivity_manager"],
                "specific": ["binance_gateway", "ftx_gateway", "coinbase_gateway"]
            },
            "fill": ["fill_service", "fill_model", "fill_simulator"],
            "risk": ["pre_trade_validator", "post_trade_reconciliation", "position_reconciliation",
                     "exchange_risk_limits"]
        },

        "utils": ["logger", "metrics", "validators", "serializers", "error_handling", "time_utils",
                  "math_utils", "config_utils", "io_utils", "concurrency_utils", "security_utils"]
    }

    # Create unit test files
    for module, submodules in unit_modules.items():
        if isinstance(submodules, list):
            module_dir = unit_dir / module
            os.makedirs(module_dir, exist_ok=True)

            for submodule in submodules:
                class_name = "".join(word.capitalize() for word in submodule.split("_"))
                create_test_file(module_dir / f"test_{submodule}.py", f"{module}.{submodule}", class_name)
        else:
            # This is a nested structure
            main_module_dir = unit_dir / module
            os.makedirs(main_module_dir, exist_ok=True)

            # Handle root files
            for submodule in submodules.get("", []):
                class_name = "".join(word.capitalize() for word in submodule.split("_"))
                create_test_file(main_module_dir / f"test_{submodule}.py", f"{module}.{submodule}", class_name)

            # Handle nested directories
            for subdir, subsubmodules in submodules.items():
                if subdir == "":
                    continue

                if isinstance(subsubmodules, list):
                    subdir_path = main_module_dir / subdir
                    os.makedirs(subdir_path, exist_ok=True)

                    for subsubmodule in subsubmodules:
                        class_name = "".join(word.capitalize() for word in subsubmodule.split("_"))
                        create_test_file(
                            subdir_path / f"test_{subsubmodule}.py",
                            f"{module}.{subdir}.{subsubmodule}",
                            class_name
                        )
                else:
                    # Another level of nesting
                    for subsubdir, subsubsubmodules in subsubmodules.items():
                        if subsubdir == "":
                            subdir_path = main_module_dir / subdir
                            os.makedirs(subdir_path, exist_ok=True)

                            for subsubmodule in subsubsubmodules:
                                class_name = "".join(word.capitalize() for word in subsubmodule.split("_"))
                                create_test_file(
                                    subdir_path / f"test_{subsubmodule}.py",
                                    f"{module}.{subdir}.{subsubmodule}",
                                    class_name
                                )
                        else:
                            subsubdir_path = main_module_dir / subdir / subsubdir
                            os.makedirs(subsubdir_path, exist_ok=True)

                            for subsubsubmodule in subsubsubmodules:
                                class_name = "".join(word.capitalize() for word in subsubsubmodule.split("_"))
                                create_test_file(
                                    subsubdir_path / f"test_{subsubsubmodule}.py",
                                    f"{module}.{subdir}.{subsubdir}.{subsubsubmodule}",
                                    class_name
                                )

    # Create integration test files
    os.makedirs(integration_dir, exist_ok=True)
    integration_tests = [
        "core_data_integration", "data_models_integration", "models_execution_integration",
        "execution_feedback_loop", "risk_system_integration", "paper_trading_flow",
        "live_trading_flow", "mode_switching_flow", "alpha_to_execution_flow",
        "market_data_to_signal_flow", "disaster_recovery_flow", "component_failure_recovery",
        "data_corruption_recovery", "exchange_outage_recovery"
    ]

    for test in integration_tests:
        create_test_file(integration_dir / f"test_{test}.py", test)

    # Create performance test directory structure
    performance_categories = {
        "latency": ["market_data_latency", "signal_generation_latency", "order_execution_latency",
                    "event_propagation_latency", "critical_path_latency"],
        "throughput": ["market_data_throughput", "event_bus_throughput", "order_throughput",
                       "signal_throughput"],
        "resource_usage": ["memory_usage", "cpu_usage", "network_usage", "storage_performance",
                           "resource_scaling"],
        "load": ["system_under_load", "market_volatility_load", "concurrent_strategies",
                 "multi_exchange_load", "backpressure_mechanism"]
    }

    for category, tests in performance_categories.items():
        category_dir = performance_dir / category
        os.makedirs(category_dir, exist_ok=True)

        for test in tests:
            create_test_file(category_dir / f"test_{test}.py", f"performance.{category}.{test}")

    # Create fixtures
    os.makedirs(fixtures_dir, exist_ok=True)
    fixtures = ["market_data_fixtures", "order_fixtures", "strategy_fixtures", "mock_exchange_fixtures"]

    for fixture in fixtures:
        create_fixture_file(fixtures_dir / f"{fixture}.py", fixture)

    # Create test utilities
    os.makedirs(utils_dir, exist_ok=True)
    utilities = ["performance_metrics", "test_data_generators", "assertions", "test_environment"]

    for utility in utilities:
        create_utility_file(utils_dir / f"{utility}.py", utility)

    print(f"\nTest directory structure created successfully in {root_dir.absolute()}")
    print(f"To run tests, use: python tests/run_tests.py")


if __name__ == "__main__":
    setup_test_structure()