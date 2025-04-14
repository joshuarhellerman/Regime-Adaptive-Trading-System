# Regime Adaptive Trading System

An institutional-grade ML-powered trading system with automatic regime detection, adaptive strategy selection, and seamless transition between paper and live trading.

## Overview

This trading system is designed with enterprise-level architecture principles to provide a robust platform for algorithmic trading across different market regimes. The system leverages machine learning for market regime detection, strategy optimization, and portfolio management, with a strong focus on risk management and system resilience.

## Key Features

- **Regime Detection & Adaptation**: Automatically identifies market regimes and adapts strategies accordingly
- **Seamless Mode Switching**: Zero-downtime transition between paper and live trading
- **ML-Powered Alpha Generation**: Advanced machine learning models for signal generation
- **Comprehensive Risk Controls**: Multi-level risk management with explicit limits and circuit breakers
- **Portfolio Optimization**: Sophisticated portfolio construction with risk constraints
- **High Performance Architecture**: Designed for low-latency trading with explicit performance budgets
- **Extensive Monitoring**: Real-time dashboards and performance analytics
- **Disaster Recovery**: Enterprise-grade reliability with transactional state management
- **Online Learning**: Continuous model improvement with proper validation safeguards

## Core Architecture Principles

- **Modularity with Clear Boundaries**: Components with well-defined interfaces and responsibilities
- **Single Source of Truth**: One authoritative source for each type of data
- **Strict Performance Budget**: Explicit latency and resource constraints for all components
- **Stateless Where Possible**: Minimizing state-dependent components for greater reliability
- **Fault Isolation**: Preventing failure cascades across system components
- **Observable System**: Comprehensive metrics and logs for all critical paths
- **Deterministic Behavior**: Consistent system responses given the same inputs
- **Online Learning**: Continuous improvement through proper feedback loops

## System Architecture

```
/Regime-Adaptive-Trading-System/
|-- config/                      # Configuration files
|-- core/                        # Core system components
|-- data/                        # Data management
|-- models/                      # ML models and strategies
|-- execution/                   # Order execution
|-- visualization/               # Dashboard and visualization
|-- ui/                          # UI components
|-- analysis/                    # Analysis modules
|-- utils/                       # Utility functions
|-- tests/                       # Tests
|-- scripts/                     # Scripts
|-- main.py                      # Entry point
|-- setup.py                     # Installation
```

## Installation

```bash
# Clone the repository
git clone https://github.com/joshuarhellerman/Regime-Adaptive-Trading-System.git
cd Regime-Adaptive-Trading-System

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

# Start the web dashboard
python -m ui.web.app

# Run backtesting
python -m analysis.backtesting_engine --config config/backtest.yaml
```

## Data Flow

### Training Flow
1. **Data Collection**: Historical and alternative data gathering
2. **Feature Engineering**: Feature transformation and selection
3. **Regime Detection**: Market state clustering and transition modeling
4. **Model Training**: Regime-specific model training and optimization
5. **Validation**: Out-of-sample performance validation
6. **Deployment**: Validated model deployment with version control

### Live Trading Flow
1. **Data Ingestion**: Real-time data streaming with quality validation
2. **Regime Classification**: Current market regime identification
3. **Signal Generation**: Strategy-based signal generation with confidence scoring
4. **Portfolio Management**: Position sizing with risk constraints
5. **Execution**: Order optimization and monitoring
6. **Feedback Loop**: Performance analysis and model adjustment

## Online Learning Architecture

The system implements a sophisticated online learning framework:

- **Multi-Environment Learning Pipeline**: Shadow environment, canary deployment, A/B testing
- **Statistical Validation Gates**: Significance testing before parameter updates
- **Hierarchical Parameter Adaptation**: System, strategy, execution, and risk parameter layers
- **Multi-Timescale Learning**: From fast execution optimization to slow architecture updates
- **Safeguards Against Overfitting**: Bayesian methods, regularization, ensemble techniques
- **Drift Detection and Adaptation**: Continuous monitoring of feature distributions

## Development Guidelines

The system follows quantitative development best practices:

- **Strict Type Enforcement**: Strong typing with runtime validation
- **Deterministic Operations**: Reproducible calculations with version control
- **Performance Profiling**: Explicit latency budgets for critical paths
- **Financial Data Handling**: Precise decimal calculations with proper time handling
- **Concurrency Control**: Clear ownership model with explicit transactions
- **Observability**: Comprehensive metrics and structured logging
- **Failure Handling**: Explicit failure modes with circuit breakers
- **Testing Methodology**: Property-based and historical replay testing

## License

### Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International

Copyright (c) 2025 Joshua R. Hellerman

This project is licensed under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-nc-sa/4.0/).

#### What this license means:

- **Attribution Required**: Anyone using this code or its derivatives must give appropriate credit to the original author, provide a link to the license, and indicate if changes were made.
- **Non-Commercial Use Only**: This code and any derivatives **may not be used for commercial purposes** without explicit written permission from the author.
- **Share-Alike Required**: If you remix, transform, or build upon this material, you must distribute your contributions under the same license as the original.
- **No Additional Restrictions**: You may not apply legal terms or technological measures that legally restrict others from doing anything the license permits.

#### Specifically prohibited without written permission:
- Using this system or its components in production trading environments
- Incorporating this code into commercial products or services
- Licensing or selling access to this system or derivative works
- Using this system to manage real money or assets for profit

#### Permitted uses:
- Academic research and educational purposes
- Personal learning and experimentation
- Non-commercial testing and development

**For commercial licensing inquiries, please contact:** [joshuahellerman@gmail.com]

## Copyright Notice

All source files in this repository contain the following copyright notice:

```
/*
 * Copyright (c) 2025 Joshua R. Hellerman
 * This code is subject to the terms of the CC BY-NC-SA 4.0 license.
 * Commercial use is prohibited without explicit written permission.
 */
```

## Contribution

Contributions are welcome for research and educational purposes. By submitting a contribution, you agree:
1. Your contributions will be licensed under the same CC BY-NC-SA 4.0 license
2. You have the right to license your contribution under these terms
3. You understand and accept the non-commercial nature of this project

## Acknowledgments

- This system architecture is inspired by institutional-grade quantitative trading systems
- Special thanks to the open-source financial and machine learning communitiesF{
