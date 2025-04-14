"""
Tests for the Fill model class in fill_model.py
"""

import unittest
from datetime import datetime
from decimal import Decimal
import uuid
from copy import deepcopy
from types import MappingProxyType

from execution.exchange.fill.fill_model import Fill


class TestFillModel(unittest.TestCase):
    """Test cases for the Fill class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.test_order_id = "order-12345"
        self.test_timestamp = datetime(2025, 4, 9, 12, 0, 0)
        self.test_instrument = "BTC-USD"
        self.test_quantity = Decimal("1.5")
        self.test_price = Decimal("50000.00")
        self.test_fees = Decimal("7.50")
        self.test_exchange_id = "exchange-001"
        self.test_exchange_fill_id = "exch-fill-12345"
        self.test_metadata = {"source": "market", "note": "test fill"}

        # Create a standard fill object for testing
        self.fill = Fill(
            order_id=self.test_order_id,
            fill_id="fill-12345",
            timestamp=self.test_timestamp,
            instrument=self.test_instrument,
            quantity=self.test_quantity,
            price=self.test_price,
            fees=self.test_fees,
            exchange_id=self.test_exchange_id,
            is_maker=True,
            exchange_fill_id=self.test_exchange_fill_id,
            metadata=self.test_metadata
        )

    def test_initialization(self):
        """Test that a Fill object can be properly initialized."""
        self.assertEqual(self.fill.order_id, self.test_order_id)
        self.assertEqual(self.fill.fill_id, "fill-12345")
        self.assertEqual(self.fill.timestamp, self.test_timestamp)
        self.assertEqual(self.fill.instrument, self.test_instrument)
        self.assertEqual(self.fill.quantity, self.test_quantity)
        self.assertEqual(self.fill.price, self.test_price)
        self.assertEqual(self.fill.fees, self.test_fees)
        self.assertEqual(self.fill.exchange_id, self.test_exchange_id)
        self.assertTrue(self.fill.is_maker)
        self.assertEqual(self.fill.exchange_fill_id, self.test_exchange_fill_id)
        self.assertEqual(dict(self.fill.metadata), self.test_metadata)

    def test_default_values(self):
        """Test that default values are properly applied when not specified."""
        minimal_fill = Fill(order_id=self.test_order_id)

        # UUID4 fill_id should be generated
        self.assertTrue(minimal_fill.fill_id)
        try:
            uuid.UUID(minimal_fill.fill_id, version=4)
            valid_uuid = True
        except ValueError:
            valid_uuid = False
        self.assertTrue(valid_uuid)

        # Default timestamp should be close to now
        time_diff = datetime.utcnow() - minimal_fill.timestamp
        self.assertLess(abs(time_diff.total_seconds()), 5)  # Within 5 seconds

        # Other defaults
        self.assertEqual(minimal_fill.instrument, "")
        self.assertEqual(minimal_fill.quantity, Decimal("0.0"))
        self.assertEqual(minimal_fill.price, Decimal("0.0"))
        self.assertEqual(minimal_fill.fees, Decimal("0.0"))
        self.assertIsNone(minimal_fill.exchange_id)
        self.assertFalse(minimal_fill.is_maker)
        self.assertIsNone(minimal_fill.exchange_fill_id)
        self.assertEqual(dict(minimal_fill.metadata), {})

    def test_type_conversion(self):
        """Test that values are properly converted to the appropriate types."""
        fill = Fill(
            order_id="test-order",
            quantity=1.5,  # float
            price=50000,  # int
            fees=7.5  # float
        )

        self.assertEqual(fill.quantity, Decimal("1.5"))
        self.assertEqual(fill.price, Decimal("50000"))
        self.assertEqual(fill.fees, Decimal("7.5"))

        # Test that strings are also properly converted
        fill2 = Fill(
            order_id="test-order",
            quantity="2.5",  # string
            price="60000.50",  # string
            fees="10.75"  # string
        )

        self.assertEqual(fill2.quantity, Decimal("2.5"))
        self.assertEqual(fill2.price, Decimal("60000.50"))
        self.assertEqual(fill2.fees, Decimal("10.75"))

    def test_immutability(self):
        """Test that Fill objects are immutable."""
        # Try to modify attributes directly
        with self.assertRaises(AttributeError):
            self.fill.order_id = "new-order-id"

        # Try to modify through _data
        with self.assertRaises(AttributeError):
            self.fill._data["order_id"] = "new-order-id"

        # Check that metadata is immutable (MappingProxyType)
        self.assertIsInstance(self.fill.metadata, MappingProxyType)
        with self.assertRaises(TypeError):
            self.fill.metadata["new_key"] = "new_value"

    def test_with_updates(self):
        """Test the with_updates method for creating modified copies."""
        new_fill = self.fill.with_updates(
            quantity=Decimal("2.5"),
            price=Decimal("55000.00")
        )

        # Check that the original is unchanged
        self.assertEqual(self.fill.quantity, self.test_quantity)
        self.assertEqual(self.fill.price, self.test_price)

        # Check that the new fill has the updated values
        self.assertEqual(new_fill.quantity, Decimal("2.5"))
        self.assertEqual(new_fill.price, Decimal("55000.00"))

        # Check that other values are preserved
        self.assertEqual(new_fill.order_id, self.test_order_id)
        self.assertEqual(new_fill.fill_id, "fill-12345")
        self.assertEqual(new_fill.timestamp, self.test_timestamp)

    def test_to_dict(self):
        """Test the to_dict method for serialization."""
        fill_dict = self.fill.to_dict()

        # Check that all keys are present
        expected_keys = [
            "order_id", "fill_id", "timestamp", "instrument", "quantity",
            "price", "fees", "exchange_id", "is_maker", "exchange_fill_id",
            "metadata"
        ]
        for key in expected_keys:
            self.assertIn(key, fill_dict)

        # Check specific conversions
        self.assertEqual(fill_dict["timestamp"], self.test_timestamp.isoformat())
        self.assertEqual(fill_dict["quantity"], float(self.test_quantity))
        self.assertEqual(fill_dict["price"], float(self.test_price))
        self.assertEqual(fill_dict["fees"], float(self.test_fees))
        self.assertEqual(fill_dict["metadata"], self.test_metadata)

    def test_from_dict(self):
        """Test the from_dict method for deserialization."""
        # Create a dictionary representing a fill
        fill_dict = {
            "order_id": "order-67890",
            "fill_id": "fill-67890",
            "timestamp": "2025-04-10T14:30:00",
            "instrument": "ETH-USD",
            "quantity": 2.75,
            "price": 3000.50,
            "fees": 5.25,
            "exchange_id": "exchange-002",
            "is_maker": False,
            "exchange_fill_id": "exch-fill-67890",
            "metadata": {"source": "limit", "note": "test deserialization"}
        }

        # Deserialize to a Fill object
        fill = Fill.from_dict(fill_dict)

        # Validate the deserialized object
        self.assertEqual(fill.order_id, "order-67890")
        self.assertEqual(fill.fill_id, "fill-67890")
        self.assertEqual(fill.timestamp, datetime(2025, 4, 10, 14, 30, 0))
        self.assertEqual(fill.instrument, "ETH-USD")
        self.assertEqual(fill.quantity, Decimal("2.75"))
        self.assertEqual(fill.price, Decimal("3000.50"))
        self.assertEqual(fill.fees, Decimal("5.25"))
        self.assertEqual(fill.exchange_id, "exchange-002")
        self.assertFalse(fill.is_maker)
        self.assertEqual(fill.exchange_fill_id, "exch-fill-67890")
        self.assertEqual(dict(fill.metadata), {"source": "limit", "note": "test deserialization"})

    def test_equality(self):
        """Test the equality comparison between Fill objects."""
        # Create an identical fill
        identical_fill = Fill(
            order_id=self.test_order_id,
            fill_id="fill-12345",
            timestamp=self.test_timestamp,
            instrument=self.test_instrument,
            quantity=self.test_quantity,
            price=self.test_price,
            fees=self.test_fees,
            exchange_id=self.test_exchange_id,
            is_maker=True,
            exchange_fill_id=self.test_exchange_fill_id,
            metadata=deepcopy(self.test_metadata)
        )

        # Create a different fill
        different_fill = Fill(
            order_id=self.test_order_id,
            fill_id="fill-67890",  # Different fill_id
            timestamp=self.test_timestamp,
            instrument=self.test_instrument,
            quantity=self.test_quantity,
            price=self.test_price,
            fees=self.test_fees,
            exchange_id=self.test_exchange_id,
            is_maker=True,
            exchange_fill_id=self.test_exchange_fill_id,
            metadata=deepcopy(self.test_metadata)
        )

        # Test equality comparisons
        self.assertEqual(self.fill, identical_fill)
        self.assertNotEqual(self.fill, different_fill)
        self.assertNotEqual(self.fill, "not a fill object")

    def test_repr(self):
        """Test the string representation of a Fill object."""
        repr_str = repr(self.fill)

        # Check that the representation contains key information
        self.assertIn(self.test_order_id, repr_str)
        self.assertIn(str(float(self.test_quantity)), repr_str)
        self.assertIn(str(float(self.test_price)), repr_str)
        self.assertIn(self.test_timestamp.isoformat(), repr_str)


if __name__ == "__main__":
    unittest.main()