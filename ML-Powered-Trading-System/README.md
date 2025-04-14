# ml_trading_system

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
git clone https://github.com/yourusername/ml_trading_system.git
cd ml_trading_system

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

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
