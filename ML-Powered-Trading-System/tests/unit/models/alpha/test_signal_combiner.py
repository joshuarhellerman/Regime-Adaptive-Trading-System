import unittest
from unittest.mock import patch, MagicMock
import time
import numpy as np

from models.alpha.signal_combiner import SignalCombiner
from models.alpha.alpha_model_interface import AlphaSignal


class TestSignalCombiner(unittest.TestCase):
    """Test cases for the SignalCombiner class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.combiner = SignalCombiner(combiner_id="test-combiner")
        
        # Create sample signals
        self.signal1 = AlphaSignal(
            instrument="BTC-USD",
            direction="long",
            strength=0.8,
            confidence=0.9,
            model_id="model1",
            timestamp=time.time()
        )
        
        self.signal2 = AlphaSignal(
            instrument="BTC-USD",
            direction="long",
            strength=0.6,
            confidence=0.7,
            model_id="model2",
            timestamp=time.time()
        )
        
        self.signal3 = AlphaSignal(
            instrument="BTC-USD",
            direction="short",
            strength=0.5,
            confidence=0.6,
            model_id="model3",
            timestamp=time.time() - 3600  # 1 hour old
        )
        
        self.signal4 = AlphaSignal(
            instrument="ETH-USD",
            direction="long",
            strength=0.7,
            confidence=0.8,
            model_id="model1",
            timestamp=time.time()
        )

    def test_initialization(self):
        """Test combiner initialization and default configuration."""
        self.assertEqual(self.combiner.combiner_id, "test-combiner")
        self.assertEqual(self.combiner.config['default_weighting'], 'confidence')
        self.assertEqual(len(self.combiner.weighting_schemes), 5)
        
    def test_configure(self):
        """Test configuration update."""
        new_config = {
            'default_weighting': 'equal',
            'minimum_confidence': 0.5,
            'model_weights': {'model1': 2.0, 'model2': 1.0}
        }
        self.combiner.configure(new_config)
        
        self.assertEqual(self.combiner.config['default_weighting'], 'equal')
        self.assertEqual(self.combiner.config['minimum_confidence'], 0.5)
        self.assertEqual(self.combiner.config['model_weights'], {'model1': 2.0, 'model2': 1.0})
        
    def test_empty_signals(self):
        """Test combining empty signal list."""
        result = self.combiner.combine_signals([])
        self.assertEqual(result, {})
        
    def test_below_minimum_signal_count(self):
        """Test case where number of signals is below minimum threshold."""
        self.combiner.config['minimum_signal_count'] = 2
        result = self.combiner.combine_signals([self.signal4])
        self.assertEqual(result, {})
        
    @patch('models.alpha.signal_combiner.EventBus')
    def test_equal_weighting(self, mock_event_bus):
        """Test equal weighting scheme."""
        result = self.combiner.combine_signals(
            [self.signal1, self.signal2], 
            weighting_scheme='equal'
        )
        
        self.assertIn('BTC-USD', result)
        combined = result['BTC-USD']
        
        # With two long signals of equal weight, direction should be long
        self.assertEqual(combined.direction, 'long')
        # Equal weighting should give average of strengths
        self.assertAlmostEqual(combined.strength, 0.7, places=1)
        # Check if event was emitted
        mock_event_bus.emit.assert_called_once()
        
    @patch('models.alpha.signal_combiner.EventBus')
    def test_confidence_weighting(self, mock_event_bus):
        """Test confidence-based weighting."""
        result = self.combiner.combine_signals(
            [self.signal1, self.signal2], 
            weighting_scheme='confidence'
        )
        
        combined = result['BTC-USD']
        # Confidence values are 0.9 and 0.7, so signal1 should have more influence
        self.assertEqual(combined.direction, 'long')
        self.assertGreater(combined.strength, 0.7)  # Should be closer to signal1's strength
        
    @patch('models.alpha.signal_combiner.EventBus')
    def test_performance_weighting(self, mock_event_bus):
        """Test performance-based weighting."""
        model_performance = {
            'model1': 0.9,
            'model2': 0.3
        }
        
        result = self.combiner.combine_signals(
            [self.signal1, self.signal2], 
            weighting_scheme='model_performance',
            model_performance=model_performance
        )
        
        combined = result['BTC-USD']
        # model1 has higher performance, so signal1 should have more influence
        self.assertEqual(combined.direction, 'long')
        self.assertGreater(combined.strength, 0.7)  # Should be closer to signal1's strength
        
    @patch('models.alpha.signal_combiner.EventBus')
    def test_performance_weighting_without_metrics(self, mock_event_bus):
        """Test performance-based weighting without performance metrics."""
        # Without performance metrics, should fall back to equal weighting
        result = self.combiner.combine_signals(
            [self.signal1, self.signal2], 
            weighting_scheme='model_performance'
        )
        
        combined = result['BTC-USD']
        self.assertEqual(combined.direction, 'long')
        self.assertAlmostEqual(combined.strength, 0.7, places=1)  # Should be average of strengths
        
    @patch('models.alpha.signal_combiner.EventBus')
    def test_time_decay_weighting(self, mock_event_bus):
        """Test time decay weighting."""
        # signal3 is 1 hour old while signal1 is fresh
        result = self.combiner.combine_signals(
            [self.signal1, self.signal3], 
            weighting_scheme='time_decay'
        )
        
        combined = result['BTC-USD']
        # signal1 (long) should have more influence than signal3 (short)
        self.assertEqual(combined.direction, 'long')
        
    @patch('models.alpha.signal_combiner.EventBus')
    def test_priority_weighting(self, mock_event_bus):
        """Test priority-based weighting."""
        self.combiner.config['model_priorities'] = {
            'model1': 3.0,
            'model2': 1.0
        }
        
        result = self.combiner.combine_signals(
            [self.signal1, self.signal2], 
            weighting_scheme='model_priority'
        )
        
        combined = result['BTC-USD']
        # model1 has higher priority, so signal1 should have more influence
        self.assertEqual(combined.direction, 'long')
        self.assertGreater(combined.strength, 0.7)  # Should be closer to signal1's strength
        
    @patch('models.alpha.signal_combiner.EventBus')
    def test_conflicting_signals(self, mock_event_bus):
        """Test combining conflicting signals."""
        result = self.combiner.combine_signals(
            [self.signal1, self.signal3], 
            weighting_scheme='equal'
        )
        
        combined = result['BTC-USD']
        # With one long and one short of equal weight, result depends on signal strength
        # signal1 has 0.8 strength (long) vs signal3 with 0.5 strength (short)
        self.assertEqual(combined.direction, 'long')
        self.assertTrue(combined.metadata['conflicting_signals'])
        
    @patch('models.alpha.signal_combiner.EventBus')
    def test_conflicting_signals_neutralization(self, mock_event_bus):
        """Test neutralization when signals conflict with similar strength."""
        # Create signals with opposite directions but similar strength
        sig_long = AlphaSignal(
            instrument="BTC-USD",
            direction="long",
            strength=0.6,
            confidence=0.7,
            model_id="model1",
            timestamp=time.time()
        )
        
        sig_short = AlphaSignal(
            instrument="BTC-USD",
            direction="short",
            strength=0.6,
            confidence=0.7,
            model_id="model2",
            timestamp=time.time()
        )
        
        # Set threshold for neutralization
        self.combiner.config['conflicting_signal_threshold'] = 0.5
        
        result = self.combiner.combine_signals(
            [sig_long, sig_short], 
            weighting_scheme='equal'
        )
        
        combined = result['BTC-USD']
        # With similar strength signals in opposite directions, should neutralize
        self.assertEqual(combined.direction, 'neutral')
        
    @patch('models.alpha.signal_combiner.EventBus')
    def test_minimum_confidence_filter(self, mock_event_bus):
        """Test filtering signals below minimum confidence."""
        # Create a signal with low confidence
        low_conf_signal = AlphaSignal(
            instrument="BTC-USD",
            direction="short",
            strength=0.9,
            confidence=0.2,  # Below default minimum of 0.3
            model_id="model4",
            timestamp=time.time()
        )
        
        result = self.combiner.combine_signals(
            [self.signal1, low_conf_signal], 
            weighting_scheme='equal'
        )
        
        combined = result['BTC-USD']
        # Only signal1 should contribute as low_conf_signal is filtered out
        self.assertEqual(combined.direction, 'long')
        self.assertAlmostEqual(combined.strength, 0.8, places=1)  # Should match signal1's strength
        
    @patch('models.alpha.signal_combiner.EventBus')
    def test_multiple_instruments(self, mock_event_bus):
        """Test combining signals for multiple instruments."""
        result = self.combiner.combine_signals(
            [self.signal1, self.signal2, self.signal4], 
            weighting_scheme='equal'
        )
        
        # Should have results for both BTC-USD and ETH-USD
        self.assertEqual(len(result), 2)
        self.assertIn('BTC-USD', result)
        self.assertIn('ETH-USD', result)
        
    def test_add_custom_weighting(self):
        """Test adding a custom weighting scheme."""
        def custom_weight_func(signals, model_performance=None):
            """Simple custom weighting function based on signal strength."""
            strengths = [signal.strength for signal in signals]
            total_strength = sum(strengths)
            return [s / total_strength for s in strengths]
        
        self.combiner.add_custom_weighting('strength_based', custom_weight_func)
        
        # Verify new scheme was added
        self.assertIn('strength_based', self.combiner.weighting_schemes)
        self.assertEqual(len(self.combiner.weighting_schemes), 6)
        
        # Test using the custom scheme
        result = self.combiner.combine_signals(
            [self.signal1, self.signal2], 
            weighting_scheme='strength_based'
        )
        
        self.assertIn('BTC-USD', result)
        combined = result['BTC-USD']
        self.assertEqual(combined.direction, 'long')
        
    def test_get_weighting_schemes(self):
        """Test retrieving available weighting scheme names."""
        schemes = self.combiner.get_weighting_schemes()
        self.assertEqual(len(schemes), 5)
        self.assertIn('equal', schemes)
        self.assertIn('confidence', schemes)
        self.assertIn('model_performance', schemes)
        self.assertIn('time_decay', schemes)
        self.assertIn('model_priority', schemes)
        
    @patch('models.alpha.signal_combiner.EventBus')
    def test_output_normalization(self, mock_event_bus):
        """Test signal strength normalization."""
        # Create signals that would produce strength > 1.0 when combined
        strong_signal1 = AlphaSignal(
            instrument="BTC-USD",
            direction="long",
            strength=0.9,
            confidence=0.9,
            model_id="model1",
            timestamp=time.time()
        )
        
        strong_signal2 = AlphaSignal(
            instrument="BTC-USD",
            direction="long",
            strength=0.9,
            confidence=0.9,
            model_id="model2",
            timestamp=time.time()
        )
        
        # With normalization on (default)
        result = self.combiner.combine_signals(
            [strong_signal1, strong_signal2], 
            weighting_scheme='equal'
        )
        combined = result['BTC-USD']
        self.assertLessEqual(combined.strength, 1.0)
        
        # With normalization off
        self.combiner.config['normalize_output'] = False
        result = self.combiner.combine_signals(
            [strong_signal1, strong_signal2], 
            weighting_scheme='equal'
        )
        combined = result['BTC-USD']
        self.assertGreaterEqual(combined.strength, 0.9)
        
    @patch('models.alpha.signal_combiner.EventBus')
    def test_unknown_weighting_scheme(self, mock_event_bus):
        """Test handling of unknown weighting scheme."""
        result = self.combiner.combine_signals(
            [self.signal1, self.signal2], 
            weighting_scheme='non_existent_scheme'
        )
        
        # Should fall back to equal weighting
        combined = result['BTC-USD']
        self.assertEqual(combined.direction, 'long')
        self.assertAlmostEqual(combined.strength, 0.7, places=1)


if __name__ == '__main__':
    unittest.main()