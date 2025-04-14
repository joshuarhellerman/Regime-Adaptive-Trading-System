#!/usr/bin/env python3
"""
ML-Powered Trading System Project Setup Script

This script creates the directory structure and placeholder files for
an institutional-grade ML-powered trading system.
"""

import os
import sys
import argparse
from pathlib import Path
import shutil

# Directory structure specification
DIRECTORY_STRUCTURE = {
    "config": {
        "files": [
            "system_config.py",
            "strategy_config.py",
            "database_config.py",
            "logging_config.py",
            "exchange_config.py",
            "trading_mode_config.py",
            "disaster_recovery_config.py"
        ]
    },
    "core": {
        "files": [
            "system.py",
            "event_bus.py",
            "state_manager.py",
            "risk_manager.py",
            "performance_metrics.py",
            "scheduler.py",
            "trading_mode_controller.py",
            "component_registry.py",
            "disaster_recovery.py",
            "health_monitor.py",
            "circuit_breaker.py"
        ]
    },
    "data": {
        "files": [
            "market_data_service.py",
            "data_integrity.py",
            "replay_service.py"
        ],
        "dirs": {
            "fetchers": {
                "files": [
                    "exchange_connector.py",
                    "historical_repository.py",
                    "alternative_data_gateway.py",
                    "mock_data_provider.py"
                ]
            },
            "processors": {
                "files": [
                    "data_normalizer.py",
                    "tick_aggregator.py",
                    "feature_engineering.py",
                    "feature_store.py"
                ]
            },
            "storage": {
                "files": [
                    "time_series_store.py",
                    "market_snapshot.py",
                    "persistent_queue.py"
                ]
            }
        }
    },
    "models": {
        "files": [],
        "dirs": {
            "alpha": {
                "files": [
                    "alpha_model_interface.py",
                    "alpha_factory.py",
                    "alpha_registry.py",
                    "alpha_result.py",
                    "signal_combiner.py"
                ]
            },
            "strategies": {
                "files": [
                    "strategy_protocol.py",
                    "strategy_context.py",
                    "breakout_strategy.py",
                    "momentum_strategy.py",
                    "mean_reversion_strategy.py",
                    "trend_following_strategy.py",
                    "volatility_strategy.py"
                ]
            },
            "regime": {
                "files": [
                    "regime_classifier.py",
                    "regime_transition_detector.py",
                    "regime_properties.py",
                    "regime_adaptor.py"
                ]
            },
            "portfolio": {
                "files": [
                    "risk_model.py",
                    "optimizer.py",
                    "allocation_manager.py",
                    "trade_generator.py"
                ]
            },
            "research": {
                "files": [
                    "research_environment.py",
                    "hyperparameter_optimization.py",
                    "feature_analyzer.py",
                    "model_validator.py",
                    "cross_validation.py"
                ]
            },
            "production": {
                "files": [
                    "model_deployer.py",
                    "prediction_service.py",
                    "model_monitoring.py"
                ]
            }
        }
    },
    "execution": {
        "files": [
            "execution_service.py"
        ],
        "dirs": {
            "order": {
                "files": [
                    "order.py",
                    "order_factory.py",
                    "order_book.py",
                    "execution_algorithm.py",
                    "twap_algorithm.py",
                    "vwap_algorithm.py",
                    "smart_router.py"
                ]
            },
            "exchange": {
                "files": [
                    "exchange_gateway.py",
                    "exchange_simulator.py",
                    "rate_limiter.py",
                    "connectivity_manager.py"
                ],
                "dirs": {
                    "exchange_specific": {
                        "files": [
                            "binance_gateway.py",
                            "ftx_gateway.py",
                            "coinbase_gateway.py"
                        ]
                    }
                }
            },
            "fill": {
                "files": [
                    "fill_service.py",
                    "fill_model.py",
                    "fill_simulator.py"
                ]
            },
            "risk": {
                "files": [
                    "pre_trade_validator.py",
                    "post_trade_reconciliation.py",
                    "position_reconciliation.py",
                    "exchange_risk_limits.py"
                ]
            }
        }
    },
    "visualization": {
        "files": [
            "unified_dashboard.py",
            "panel_registry.py",
            "data_formatter.py"
        ],
        "dirs": {
            "panels": {
                "files": [
                    "base_panel.py",
                    "performance_panel.py",
                    "portfolio_panel.py",
                    "regime_panel.py",
                    "strategy_panel.py",
                    "execution_panel.py",
                    "system_health_panel.py"
                ]
            },
            "adapters": {
                "files": [
                    "ui_adapter.py"
                ]
            },
            "plots": {
                "files": [
                    "equity_plots.py",
                    "feature_plots.py",
                    "regime_plots.py",
                    "risk_plots.py",
                    "correlation_plots.py"
                ]
            }
        }
    },
    "ui": {
        "files": [],
        "dirs": {
            "web": {
                "files": [
                    "app.py",
                    "api.py",
                    "auth.py",
                    "web_socket_handler.py",
                    "routes.py"
                ]
            },
            "api": {
                "files": [
                    "paper_trading_api.py",
                    "trading_mode_api.py",
                    "system_config_api.py",
                    "model_management_api.py"
                ]
            },
            "static": {
                "dirs": {
                    "js": {"files": []},
                    "css": {"files": []},
                    "images": {"files": []}
                }
            },
            "templates": {
                "files": [
                    "index.html",
                    "dashboard.html",
                    "settings.html",
                    "login.html",
                    "system_status.html"
                ]
            },
            "components": {
                "files": [
                    "navigation.py",
                    "forms.py",
                    "modals.py",
                    "charts.py",
                    "tables.py"
                ]
            }
        }
    },
    "analysis": {
        "files": [
            "performance_analyzer.py",
            "risk_analyzer.py",
            "strategy_analyzer.py",
            "regime_analyzer.py",
            "market_calibration.py",
            "live_system_evaluator.py",
            "backtesting_engine.py",
            "comparison_framework.py",
            "attribution_analysis.py",
            "scenario_simulator.py",
            "optimization_analyzer.py"
        ]
    },
    "utils": {
        "files": [
            "logger.py",
            "metrics.py",
            "validators.py",
            "serializers.py",
            "error_handling.py",
            "time_utils.py",
            "math_utils.py",
            "config_utils.py",
            "io_utils.py",
            "concurrency_utils.py",
            "security_utils.py"
        ]
    },
    "tests": {
        "dirs": {
            "unit": {"files": []},
            "integration": {"files": []},
            "performance": {"files": []}
        }
    },
    "scripts": {
        "files": [
            "setup_database.py",
            "download_historical.py",
            "optimize_system.py"
        ]
    }
}

# Python module template with docstring
MODULE_TEMPLATE = '''"""
{module_name}

{module_description}
"""

# Imports

# Constants

# Classes/Functions

def main():
    """Main function."""
    pass

if __name__ == "__main__":
    main()
'''


def create_directory_structure(base_dir, structure, current_path=""):
    """Recursively create directory structure and placeholder files."""
    for dir_name, contents in structure.items():
        dir_path = os.path.join(base_dir, current_path, dir_name)

        # Create directory
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created directory: {dir_path}")

        # Create files
        if "files" in contents:
            for file_name in contents["files"]:
                file_path = os.path.join(dir_path, file_name)

                # Skip if file already exists
                if os.path.exists(file_path):
                    print(f"File already exists (skipping): {file_path}")
                    continue

                # Generate module description
                module_name = os.path.splitext(file_name)[0].replace("_", " ").title()

                # Generate a meaningful description based on the file name
                descriptions = {
                    "system": "Main system controller and initialization",
                    "event_bus": "Event distribution and handling system",
                    "state_manager": "Manages system state with transaction support",
                    "risk_manager": "Manages trading risk and exposure limits",
                    "performance_metrics": "Calculates and tracks system performance metrics",
                    "scheduler": "Schedules tasks with precise timing",
                    "trading_mode_controller": "Controls transitions between paper and live trading",
                    "component_registry": "Registry for system components with lifecycle management",
                    "disaster_recovery": "System recovery and backup mechanisms",
                    "health_monitor": "Monitors system health and performance",
                    "circuit_breaker": "Trading circuit breakers with configurable thresholds",
                    "market_data_service": "Centralizes access to all market data",
                    "alpha_model_interface": "Interface for alpha model implementation",
                    "strategy_protocol": "Protocol definition for trading strategies"
                }

                # Generate default description if not in our mapping
                module_simple_name = os.path.splitext(file_name)[0]
                module_description = descriptions.get(
                    module_simple_name,
                    f"Implementation of {module_name}"
                )

                # Create file with template
                with open(file_path, "w") as f:
                    content = MODULE_TEMPLATE.format(
                        module_name=module_name,
                        module_description=module_description
                    )
                    f.write(content)

                print(f"Created file: {file_path}")

        # Process subdirectories
        if "dirs" in contents:
            new_path = os.path.join(current_path, dir_name)
            create_directory_structure(base_dir, contents["dirs"], new_path)


def create_main_file(base_dir):
    """Create the main.py file with system initialization."""
    main_content = '''"""
Main Entry Point

This is the main entry point for the ML-Powered Trading System.
It initializes all components and starts the system.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from config.system_config import load_config
from core.system import System
from utils.logger import setup_logging

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="ML-Powered Trading System")
    parser.add_argument("--config", type=str, default="config/default.yaml",
                       help="Path to configuration file")
    parser.add_argument("--mode", type=str, choices=["paper", "live"],
                       default="paper", help="Trading mode")
    parser.add_argument("--log-level", type=str, 
                       choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                       default="INFO", help="Logging level")
    return parser.parse_args()

def main():
    """Main function to start the trading system."""
    args = parse_arguments()

    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    logger.info(f"Starting ML-Powered Trading System in {args.mode} mode")

    try:
        # Load configuration
        config = load_config(args.config)

        # Initialize and start the system
        system = System(config, mode=args.mode)
        system.initialize()
        system.start()

        # Keep the main thread running
        system.join()

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, shutting down")
        if 'system' in locals():
            system.shutdown()
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}", exc_info=True)
        if 'system' in locals():
            system.emergency_shutdown()
        sys.exit(1)

    logger.info("System shutdown complete")

if __name__ == "__main__":
    main()
'''

    main_path = os.path.join(base_dir, "main.py")
    with open(main_path, "w") as f:
        f.write(main_content)
    print(f"Created main entry point: {main_path}")


def create_setup_file(base_dir, project_name):
    """Create setup.py file for the project."""
    setup_content = f'''"""
Setup file for {project_name}
"""

from setuptools import setup, find_packages

setup(
    name="{project_name}",
    version="0.1.0",
    description="ML-Powered Trading System",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "pyyaml",
        "requests",
        "websockets",
        "fastapi",
        "uvicorn",
        "pytest",
        "pyarrow",
        "dash",
        "plotly",
        "xgboost",
        "lightgbm",
        "ccxt",
        "joblib",
    ],
    python_requires=">=3.8",
)
'''

    setup_path = os.path.join(base_dir, "setup.py")
    with open(setup_path, "w") as f:
        f.write(setup_content)
    print(f"Created setup file: {setup_path}")


def create_readme(base_dir, project_name):
    """Create README.md file for the project."""
    readme_content = f'''# {project_name}

An institutional-grade ML-powered trading system with paper and live trading capabilities.

## Features

- Unified interface for paper and live trading
- Advanced ML models for alpha generation and regime detection
- Comprehensive risk management
- Real-time performance monitoring
- Disaster recovery capabilities
- Interactive dashboard for system monitoring
- Extensive configuration options

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/{project_name}.git
cd {project_name}

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate

# Install dependencies
pip install -e .
```

## Usage

```bash
# Start the system in paper trading mode
python main.py --mode paper

# Start the system in live trading mode
python main.py --mode live --config config/live.yaml
```

## Project Structure

- `config/`: Configuration files
- `core/`: Core system components
- `data/`: Data management
- `models/`: ML models and strategies
- `execution/`: Order execution
- `visualization/`: Dashboards and visualization
- `ui/`: User interface components
- `analysis/`: Analysis tools
- `utils/`: Utility functions
- `tests/`: Tests
- `scripts/`: Helper scripts

## License

This project is licensed under the MIT License - see the LICENSE file for details.
'''

    readme_path = os.path.join(base_dir, "README.md")
    with open(readme_path, "w") as f:
        f.write(readme_content)
    print(f"Created README: {readme_path}")


def create_gitignore(base_dir):
    """Create .gitignore file for the project."""
    gitignore_content = '''# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# C extensions
*.so

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Unit test / coverage reports
htmlcov/
.tox/
.nox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
.hypothesis/
.pytest_cache/

# Jupyter Notebook
.ipynb_checkpoints

# Environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# IDE specific files
.idea/
.vscode/
*.swp
*.swo

# Project specific
*.log
data/historical/
data/cache/
config/secrets.yaml
credentials.json
'''

    gitignore_path = os.path.join(base_dir, ".gitignore")
    with open(gitignore_path, "w") as f:
        f.write(gitignore_content)
    print(f"Created .gitignore: {gitignore_path}")


def main():
    parser = argparse.ArgumentParser(description="Set up ML-Powered Trading System project structure")
    parser.add_argument("--dir", type=str, default="ML-Powered-Trading-System",
                        help="Base directory for the project")
    parser.add_argument("--name", type=str, default="ml_trading_system",
                        help="Project package name")
    args = parser.parse_args()

    base_dir = args.dir
    project_name = args.name

    # Check if directory exists
    if os.path.exists(base_dir):
        response = input(
            f"Directory {base_dir} already exists. Do you want to continue and potentially overwrite files? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            sys.exit(0)
    else:
        os.makedirs(base_dir)

    # Create directory structure
    create_directory_structure(base_dir, DIRECTORY_STRUCTURE)

    # Create main.py
    create_main_file(base_dir)

    # Create setup.py
    create_setup_file(base_dir, project_name)

    # Create README.md
    create_readme(base_dir, project_name)

    # Create .gitignore
    create_gitignore(base_dir)

    print(f"\nProject structure created successfully in: {os.path.abspath(base_dir)}")
    print(f"To get started, navigate to the project directory:")
    print(f"cd {base_dir}")
    print("And initialize a git repository:")
    print("git init")
    print("git add .")
    print("git commit -m 'Initial project structure'")


if __name__ == "__main__":
    main()