import unittest
from unittest.mock import patch, MagicMock
import time
import pandas as pd
import numpy as np
from typing import Dict, List, Set, Any

from models.alpha.alpha_model_interface import AlphaSignal, AlphaModel
from core.event_bus import EventBus


class TestAlphaSignal(unittest.TestCase):
    """Tests for the AlphaSignal class."""

    def setUp(self):
        self.signal = AlphaSignal(
            instrument="AAPL",
            direction="long",
            strength=0.75,
            confidence=0.8,
            model_id="test-model-001",
            timestamp=1617283200.0,
            signal_id="test-signal-001",
            metadata={"source": "price_momentum"},
            expiration=1617369600.0  # 24h later
        )

    def test_init_default_values(self):
        """Test initialization with default values."""
        current_time = time.time()
        
        # Create signal with minimal parameters
        signal = AlphaSignal(
            instrument="MSFT",
            direction="short",
            strength=-0.5, 
            confidence=0.6,
            model_id="test-model"
        )
        
        # Check default values
        self.assertEqual(signal.instrument, "MSFT")
        self.assertEqual(signal.direction, "short")
        self.assertEqual(signal.strength, -0.5)
        self.assertEqual(signal.confidence, 0.6)
        self.assertEqual(signal.model_id, "test-model")
        self.assertIsNotNone(signal.timestamp)
        self.assertGreaterEqual(signal.timestamp, current_time - 1)
        self.assertLessEqual(signal.timestamp, current_time + 1)
        self.assertTrue(signal.signal_id.startswith("signal-"))
        self.assertEqual(signal.metadata, {})
        self.assertAlmostEqual(signal.expiration, signal.timestamp + 86400, delta=1)

    def test_init_validation(self):
        """Test validation during initialization."""
        # Invalid direction
        with self.assertRaises(ValueError):
            AlphaSignal(
                instrument="AAPL",
                direction="invalid",
                strength=0.5,
                confidence=0.7,
                model_id="test-model"
            )
        
        # Invalid strength (too high)
        with self.assertRaises(ValueError):
            AlphaSignal(
                instrument="AAPL",
                direction="long",
                strength=1.5,
                confidence=0.7,
                model_id="test-model"
            )
        
        # Invalid strength (too low)
        with self.assertRaises(ValueError):
            AlphaSignal(
                instrument="AAPL",
                direction="long",
                strength=-1.5,
                confidence=0.7,
                model_id="test-model"
            )
        
        # Invalid confidence (too high)
        with self.assertRaises(ValueError):
            AlphaSignal(
                instrument="AAPL",
                direction="long",
                strength=0.5,
                confidence=1.7,
                model_id="test-model"
            )
        
        # Invalid confidence (too low)
        with self.assertRaises(ValueError):
            AlphaSignal(
                instrument="AAPL",
                direction="long",
                strength=0.5,
                confidence=-0.3,
                model_id="test-model"
            )

    def test_is_expired(self):
        """Test is_expired method."""
        # Not expired
        with patch('time.time', return_value=1617283200.0):
            self.assertFalse(self.signal.is_expired())
        
        # Expired
        with patch('time.time', return_value=1617369601.0):
            self.assertTrue(self.signal.is_expired())

    def test_adjusted_strength(self):
        """Test adjusted_strength method."""
        self.assertAlmostEqual(self.signal.adjusted_strength(), 0.75 * 0.8)
        
        # Test with different values
        signal = AlphaSignal(
            instrument="MSFT",
            direction="short",
            strength=-0.4,
            confidence=0.5,
            model_id="test-model"
        )
        self.assertAlmostEqual(signal.adjusted_strength(), -0.4 * 0.5)

    def test_to_dict(self):
        """Test to_dict method."""
        signal_dict = self.signal.to_dict()
        
        expected_dict = {
            'instrument': 'AAPL',
            'direction': 'long',
            'strength': 0.75,
            'confidence': 0.8,
            'model_id': 'test-model-001',
            'timestamp': 1617283200.0,
            'signal_id': 'test-signal-001',
            'metadata': {'source': 'price_momentum'},
            'expiration': 1617369600.0
        }
        
        self.assertEqual(signal_dict, expected_dict)

    def test_from_dict(self):
        """Test from_dict class method."""
        signal_dict = {
            'instrument': 'AAPL',
            'direction': 'long',
            'strength': 0.75,
            'confidence': 0.8,
            'model_id': 'test-model-001',
            'timestamp': 1617283200.0,
            'signal_id': 'test-signal-001',
            'metadata': {'source': 'price_momentum'},
            'expiration': 1617369600.0
        }
        
        signal = AlphaSignal.from_dict(signal_dict)
        
        self.assertEqual(signal.instrument, 'AAPL')
        self.assertEqual(signal.direction, 'long')
        self.assertEqual(signal.strength, 0.75)
        self.assertEqual(signal.confidence, 0.8)
        self.assertEqual(signal.model_id, 'test-model-001')
        self.assertEqual(signal.timestamp, 1617283200.0)
        self.assertEqual(signal.signal_id, 'test-signal-001')
        self.assertEqual(signal.metadata, {'source': 'price_momentum'})
        self.assertEqual(signal.expiration, 1617369600.0)

    def test_repr(self):
        """Test __repr__ method."""
        expected_repr = "AlphaSignal(instrument=AAPL, direction=long, strength=0.75, confidence=0.80)"
        self.assertEqual(repr(self.signal), expected_repr)


class ConcreteAlphaModel(AlphaModel):
    """Concrete implementation of AlphaModel for testing."""
    
    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> List[AlphaSignal]:
        """Generate test signals based on input data."""
        signals = []
        for instrument, df in data.items():
            # Simple logic: positive returns → long, negative returns → short
            if 'return' in df.columns and len(df) > 0:
                last_return = df['return'].iloc[-1]
                if last_return > 0:
                    signals.append(AlphaSignal(
                        instrument=instrument,
                        direction="long",
                        strength=min(last_return, 1.0),
                        confidence=0.7,
                        model_id=self.model_id
                    ))
                elif last_return < 0:
                    signals.append(AlphaSignal(
                        instrument=instrument,
                        direction="short",
                        strength=max(last_return, -1.0),
                        confidence=0.7,
                        model_id=self.model_id
                    ))
        return signals
    
    def get_required_features(self) -> Dict[str, Set[str]]:
        """Return features required by this model."""
        return {
            'equity': {'open', 'high', 'low', 'close', 'volume', 'return'},
            'crypto': {'open', 'high', 'low', 'close', 'volume', 'return'}
        }


class TestAlphaModel(unittest.TestCase):
    """Tests for the AlphaModel base class using ConcreteAlphaModel."""

    def setUp(self):
        # Create test model
        self.model = ConcreteAlphaModel(model_id="test-model", parameters={"lookback": 10})
        
        # Mock EventBus
        self.event_bus_mock = MagicMock()
        EventBus.emit = self.event_bus_mock
        
        # Create test data
        self.test_data = {
            'AAPL': pd.DataFrame({
                'open': [150.0, 151.0, 153.0],
                'high': [155.0, 156.0, 157.0],
                'low': [149.0, 150.0, 151.0],
                'close': [154.0, 155.0, 156.0],
                'volume': [1000000, 1100000, 1200000],
                'return': [0.01, 0.02, 0.03]
            }),
            'MSFT': pd.DataFrame({
                'open': [250.0, 248.0, 247.0],
                'high': [255.0, 253.0, 252.0],
                'low': [248.0, 246.0, 245.0],
                'close': [253.0, 250.0, 248.0],
                'volume': [900000, 950000, 930000],
                'return': [0.01, -0.01, -0.02]
            })
        }

    def test_init(self):
        """Test initialization."""
        # Test with provided model_id
        model = ConcreteAlphaModel(model_id="custom-id", parameters={"lookback": 5})
        self.assertEqual(model.model_id, "custom-id")
        self.assertEqual(model.parameters, {"lookback": 5})
        self.assertEqual(model.last_update_time, 0)
        self.assertEqual(model.signals, {})
        
        # Test with default model_id
        model = ConcreteAlphaModel()
        self.assertTrue(model.model_id.startswith("alpha-"))
        self.assertEqual(model.parameters, {})

    def test_update(self):
        """Test update method."""
        with patch('time.time', return_value=1617283200.0):
            signals = self.model.update(self.test_data)
        
        # Check that signals were generated
        self.assertEqual(len(signals), 2)
        
        # Check that signals were added to model's signals dict
        self.assertEqual(len(self.model.signals), 2)
        
        # Check that EventBus.emit was called for each signal
        self.assertEqual(self.event_bus_mock.call_count, 2)
        
        # Check last_update_time was updated
        self.assertEqual(self.model.last_update_time, 1617283200.0)

    def test_get_active_signals(self):
        """Test get_active_signals method."""
        # Add signals
        self.model.update(self.test_data)
        
        # Test active signals
        signals = self.model.get_active_signals()
        self.assertEqual(len(signals), 2)
        
        # Make signals expire
        with patch('time.time', return_value=time.time() + 100000):
            signals = self.model.get_active_signals()
            self.assertEqual(len(signals), 0)

    def test_get_signal(self):
        """Test get_signal method."""
        # Add signals
        signals = self.model.update(self.test_data)
        signal_id = signals[0].signal_id
        
        # Get existing signal
        signal = self.model.get_signal(signal_id)
        self.assertEqual(signal.signal_id, signal_id)
        
        # Get non-existent signal
        signal = self.model.get_signal("non-existent")
        self.assertIsNone(signal)

    def test_get_instrument_signals(self):
        """Test get_instrument_signals method."""
        # Add signals
        self.model.update(self.test_data)
        
        # Get signals for specific instrument
        aapl_signals = self.model.get_instrument_signals("AAPL")
        self.assertEqual(len(aapl_signals), 1)
        self.assertEqual(aapl_signals[0].instrument, "AAPL")
        
        msft_signals = self.model.get_instrument_signals("MSFT")
        self.assertEqual(len(msft_signals), 1)
        self.assertEqual(msft_signals[0].instrument, "MSFT")
        
        # Get signals for non-existent instrument
        goog_signals = self.model.get_instrument_signals("GOOG")
        self.assertEqual(len(goog_signals), 0)

    def test_clear_signals(self):
        """Test clear_signals method."""
        # Add signals
        self.model.update(self.test_data)
        self.assertEqual(len(self.model.signals), 2)
        
        # Clear signals
        count = self.model.clear_signals()
        self.assertEqual(count, 2)
        self.assertEqual(len(self.model.signals), 0)

    def test_clean_expired_signals(self):
        """Test _clean_expired_signals method."""
        # Add signals
        self.model.update(self.test_data)
        self.assertEqual(len(self.model.signals), 2)
        
        # Make signals expire and clean them
        with patch('time.time', return_value=time.time() + 100000):
            self.model._clean_expired_signals()
            self.assertEqual(len(self.model.signals), 0)

    def test_get_parameters(self):
        """Test get_parameters method."""
        # Get parameters
        params = self.model.get_parameters()
        self.assertEqual(params, {"lookback": 10})
        
        # Check that the returned object is a copy
        params["new_param"] = "test"
        self.assertNotIn("new_param", self.model.parameters)

    def test_set_parameters(self):
        """Test set_parameters method."""
        # Set new parameters
        self.model.set_parameters({"lookback": 20, "threshold": 0.05})
        
        # Check that parameters were updated
        self.assertEqual(self.model.parameters, {"lookback": 20, "threshold": 0.05})
        
        # Check that EventBus.emit was called
        self.event_bus_mock.assert_called_with(
            "alpha.parameter_update",
            {
                'model_id': "test-model",
                'parameters': {"lookback": 20, "threshold": 0.05},
                'timestamp': self.event_bus_mock.call_args[0][1]['timestamp']
            }
        )

    def test_get_model_info(self):
        """Test get_model_info method."""
        # Update model to set last_update_time
        with patch('time.time', return_value=1617283200.0):
            self.model.update(self.test_data)
        
        # Get model info
        info = self.model.get_model_info()
        
        # Check info
        self.assertEqual(info['model_id'], "test-model")
        self.assertEqual(info['model_type'], "ConcreteAlphaModel")
        self.assertEqual(info['parameters'], {"lookback": 10})
        self.assertEqual(info['last_update_time'], 1617283200.0)
        self.assertEqual(info['active_signals_count'], 2)


if __name__ == '__main__':
    unittest.main()