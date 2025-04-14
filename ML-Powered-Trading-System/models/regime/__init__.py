"""
Market Regime Classification and Adaptation Framework

This module provides tools for identifying market regimes,
detecting transitions between regimes, calculating regime properties,
and adapting trading strategy parameters based on the current regime.

Components:
- online_regime_clusterer.py: Incremental clustering for regime detection
- regime_transition_detector.py: Detects regime transitions using change point detection
- regime_properties.py: Calculates and tracks statistical properties of regimes
- regime_adaptor.py: Adapts strategy parameters based on regime properties
- regime_classifier.py: Main interface that orchestrates all components

Usage:
```python
from models.regime import RegimeClassifier

# Initialize the classifier
classifier = RegimeClassifier(
    model_id="btc_usd_hourly",
    feature_cols=["returns", "volatility", "rsi", "adx", "volume"],
    window_size=500
)

# Process new market data
result = classifier.process(market_data_df, performance_metrics=performance)

# Get the current regime and adapted parameters
current_regime = result['regime']
adapted_params = result['adapted_parameters']

# Apply adapted parameters to your strategy
strategy.update_parameters(adapted_params)
```
"""

# Import main components
from .regime_classifier import RegimeClassifier
from .online_regime_clusterer import OnlineRegimeClusterer
from .regime_transition_detector import RegimeTransitionDetector
from .regime_properties import RegimePropertiesCalculator
from .regime_adaptor import RegimeAdaptor

__all__ = [
    'RegimeClassifier',
    'OnlineRegimeClusterer',
    'RegimeTransitionDetector',
    'RegimePropertiesCalculator',
    'RegimeAdaptor'
]