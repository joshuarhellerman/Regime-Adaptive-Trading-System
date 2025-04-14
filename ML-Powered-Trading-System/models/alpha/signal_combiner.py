"""
Signal Combiner Module

This module provides utilities for combining signals from multiple alpha models
into a unified alpha signal for each instrument, using customizable weighting schemes.
"""

import logging
from typing import List, Dict, Any, Optional, Callable
import time
import numpy as np

from models.alpha.alpha_model_interface import AlphaSignal
from core.event_bus import EventBus

# Configure logger
logger = logging.getLogger(__name__)

class SignalCombiner:
    """
    Utility for combining multiple alpha signals into a unified signal.

    This class provides methods for aggregating signals from different alpha models,
    using various weighting schemes and combination strategies to produce a final
    actionable signal for each instrument.
    """

    def __init__(self, combiner_id: str = None):
        """
        Initialize a new signal combiner.

        Args:
            combiner_id: Unique identifier for this combiner instance
        """
        self.combiner_id = combiner_id or f"combiner-{int(time.time())}"
        self.weighting_schemes = {
            'equal': self._equal_weighting,
            'confidence': self._confidence_weighting,
            'model_performance': self._performance_weighting,
            'time_decay': self._time_decay_weighting,
            'model_priority': self._priority_weighting
        }

        # Default configuration
        self.config = {
            'default_weighting': 'confidence',
            'time_decay_half_life': 86400,  # 24 hours in seconds
            'minimum_confidence': 0.3,
            'minimum_signal_count': 1,
            'conflicting_signal_threshold': 0.3,
            'normalize_output': True,
            'model_weights': {},
            'model_priorities': {}
        }

        logger.info(f"Initialized signal combiner '{self.combiner_id}'")

    def configure(self, config: Dict[str, Any]) -> None:
        """
        Configure the signal combiner.

        Args:
            config: Configuration parameters
        """
        self.config.update(config)
        logger.info(f"Updated configuration for signal combiner '{self.combiner_id}'")

    def combine_signals(self, signals: List[AlphaSignal],
                      weighting_scheme: str = None,
                      model_performance: Dict[str, float] = None) -> Dict[str, AlphaSignal]:
        """
        Combine multiple alpha signals into a single signal per instrument.

        Args:
            signals: List of alpha signals to combine
            weighting_scheme: Name of weighting scheme to use (default: from config)
            model_performance: Optional performance metrics for models

        Returns:
            Dict mapping instrument IDs to combined signals
        """
        if not signals:
            return {}

        # Group signals by instrument
        signals_by_instrument = {}
        for signal in signals:
            if signal.instrument not in signals_by_instrument:
                signals_by_instrument[signal.instrument] = []
            signals_by_instrument[signal.instrument].append(signal)

        # Use default weighting scheme if not specified
        if not weighting_scheme:
            weighting_scheme = self.config['default_weighting']

        # Check if weighting scheme exists
        if weighting_scheme not in self.weighting_schemes:
            logger.warning(f"Unknown weighting scheme '{weighting_scheme}', falling back to 'equal'")
            weighting_scheme = 'equal'

        # Combine signals for each instrument
        combined_signals = {}
        for instrument, instrument_signals in signals_by_instrument.items():
            # Skip if not enough signals
            if len(instrument_signals) < self.config['minimum_signal_count']:
                logger.debug(f"Not enough signals for instrument {instrument} (need {self.config['minimum_signal_count']})")
                continue

            # Get weighting function
            weighting_func = self.weighting_schemes[weighting_scheme]

            # Calculate weights
            weights = weighting_func(instrument_signals, model_performance)

            # Apply minimum confidence filter
            valid_signals = []
            valid_weights = []
            for signal, weight in zip(instrument_signals, weights):
                if signal.confidence >= self.config['minimum_confidence']:
                    valid_signals.append(signal)
                    valid_weights.append(weight)

            # Skip if no valid signals after filtering
            if not valid_signals:
                logger.debug(f"No valid signals for instrument {instrument} after confidence filtering")
                continue

            # Normalize weights if any remain
            if valid_weights:
                weight_sum = sum(valid_weights)
                if weight_sum > 0:
                    valid_weights = [w / weight_sum for w in valid_weights]

            # Combine signals
            combined_signal = self._compute_combined_signal(instrument, valid_signals, valid_weights)

            if combined_signal:
                combined_signals[instrument] = combined_signal

                # Emit combined signal event
                EventBus.emit("alpha.combined_signal", {
                    'combiner_id': self.combiner_id,
                    'instrument': instrument,
                    'signal': combined_signal.to_dict(),
                    'source_count': len(valid_signals),
                    'timestamp': time.time()
                })

        return combined_signals

    def _compute_combined_signal(self, instrument: str, signals: List[AlphaSignal],
                              weights: List[float]) -> Optional[AlphaSignal]:
        """
        Compute a combined signal from weighted individual signals.

        Args:
            instrument: Instrument ID
            signals: List of signals for this instrument
            weights: Corresponding weights for each signal

        Returns:
            Combined AlphaSignal or None if no valid combination could be made
        """
        # Compute weighted strength
        weighted_strength = 0.0
        for signal, weight in zip(signals, weights):
            if signal.direction == 'long':
                weighted_strength += signal.strength * weight
            elif signal.direction == 'short':
                weighted_strength -= signal.strength * weight

        # Determine direction based on weighted strength
        if abs(weighted_strength) < self.config['conflicting_signal_threshold']:
            direction = 'neutral'
        else:
            direction = 'long' if weighted_strength > 0 else 'short'

        # Normalize strength if requested
        strength = weighted_strength
        if self.config['normalize_output']:
            strength = max(-1.0, min(1.0, weighted_strength))

        # Compute confidence as weighted average of source confidences
        confidence = 0.0
        for signal, weight in zip(signals, weights):
            confidence += signal.confidence * weight

        # Create metadata with source signals
        metadata = {
            'source_signals': [s.signal_id for s in signals],
            'source_models': [s.model_id for s in signals],
            'combination_method': 'weighted_average',
            'weights': weights,
            'conflicting_signals': any(
                s1.direction != s2.direction and s1.direction != 'neutral' and s2.direction != 'neutral'
                for s1 in signals for s2 in signals
            )
        }

        # Create combined signal
        return AlphaSignal(
            instrument=instrument,
            direction=direction,
            strength=abs(strength),
            confidence=confidence,
            model_id=self.combiner_id,
            metadata=metadata
        )

    def _equal_weighting(self, signals: List[AlphaSignal],
                       model_performance: Dict[str, float] = None) -> List[float]:
        """
        Apply equal weighting to all signals.

        Args:
            signals: List of signals
            model_performance: Not used in this scheme

        Returns:
            List of equal weights
        """
        return [1.0 / len(signals)] * len(signals)

    def _confidence_weighting(self, signals: List[AlphaSignal],
                           model_performance: Dict[str, float] = None) -> List[float]:
        """
        Weight signals by their confidence values.

        Args:
            signals: List of signals
            model_performance: Not used in this scheme

        Returns:
            List of weights based on signal confidence
        """
        confidences = [signal.confidence for signal in signals]
        total_confidence = sum(confidences)

        if total_confidence > 0:
            return [conf / total_confidence for conf in confidences]
        else:
            return [1.0 / len(signals)] * len(signals)

    def _performance_weighting(self, signals: List[AlphaSignal],
                            model_performance: Dict[str, float] = None) -> List[float]:
        """
        Weight signals by model performance metrics.

        Args:
            signals: List of signals
            model_performance: Dict mapping model IDs to performance metrics

        Returns:
            List of weights based on model performance
        """
        if not model_performance:
            return self._equal_weighting(signals)

        # Get performance value for each signal's model
        performances = []
        for signal in signals:
            perf = model_performance.get(signal.model_id, 0.0)
            performances.append(max(0.1, perf))  # Ensure minimum weight

        # Normalize to sum to 1
        total_performance = sum(performances)
        if total_performance > 0:
            return [perf / total_performance for perf in performances]
        else:
            return [1.0 / len(signals)] * len(signals)

    def _time_decay_weighting(self, signals: List[AlphaSignal],
                           model_performance: Dict[str, float] = None) -> List[float]:
        """
        Weight signals by recency, with exponential decay over time.

        Args:
            signals: List of signals
            model_performance: Not used in this scheme

        Returns:
            List of weights based on signal age
        """
        current_time = time.time()
        half_life = self.config['time_decay_half_life']

        # Calculate weight for each signal based on age
        weights = []
        for signal in signals:
            age = current_time - signal.timestamp
            weight = np.exp(-age * np.log(2) / half_life)
            weights.append(weight)

        # Normalize to sum to 1
        total_weight = sum(weights)
        if total_weight > 0:
            return [w / total_weight for w in weights]
        else:
            return [1.0 / len(signals)] * len(signals)

    def _priority_weighting(self, signals: List[AlphaSignal],
                         model_performance: Dict[str, float] = None) -> List[float]:
        """
        Weight signals by model priority settings.

        Args:
            signals: List of signals
            model_performance: Not used in this scheme

        Returns:
            List of weights based on model priorities
        """
        model_priorities = self.config.get('model_priorities', {})

        # If no priorities set, use model weights if available
        if not model_priorities and 'model_weights' in self.config:
            model_priorities = self.config['model_weights']

        # Get priority for each signal's model
        priorities = []
        for signal in signals:
            priority = model_priorities.get(signal.model_id, 1.0)
            priorities.append(max(0.0, priority))

        # Normalize to sum to 1
        total_priority = sum(priorities)
        if total_priority > 0:
            return [p / total_priority for p in priorities]
        else:
            return [1.0 / len(signals)] * len(signals)

    def add_custom_weighting(self, name: str, weighting_func: Callable) -> None:
        """
        Add a custom weighting scheme.

        Args:
            name: Name of the weighting scheme
            weighting_func: Function that takes signals and returns weights
        """
        self.weighting_schemes[name] = weighting_func
        logger.info(f"Added custom weighting scheme '{name}' to combiner '{self.combiner_id}'")

    def get_weighting_schemes(self) -> List[str]:
        """
        Get list of available weighting schemes.

        Returns:
            List of weighting scheme names
        """
        return list(self.weighting_schemes.keys())