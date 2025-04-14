"""
tests/utils/test_serializers.py - Tests for serializers

This module contains tests for the serializers utility functions and classes.
"""

import json
import unittest
from datetime import datetime, date
from decimal import Decimal
from enum import Enum
from unittest.mock import patch, Mock

import pytest

from utils.serializers import (
    JSONEncoder,
    serialize_to_json,
    deserialize_from_json,
    serialize_to_bson,
    deserialize_from_bson,
    serialize_to_msgpack,
    deserialize_from_msgpack,
    serialize_to_avro,
    deserialize_from_avro,
    serialize_to_protobuf,
    deserialize_from_protobuf,
    SerializationError,
    DeserializationError
)


class TestEnum(Enum):
    """Test enum for serialization tests"""
    OPTION1 = "option1"
    OPTION2 = "option2"


class TestJSONEncoder(unittest.TestCase):
    """Test the custom JSONEncoder class"""

    def test_encode_datetime(self):
        """Test encoding datetime objects"""
        dt = datetime(2023, 5, 15, 12, 30, 45)
        encoder = JSONEncoder()
        result = encoder.encode({"timestamp": dt})
        expected = '{"timestamp": "2023-05-15T12:30:45"}'
        self.assertEqual(json.loads(result), json.loads(expected))

    def test_encode_date(self):
        """Test encoding date objects"""
        d = date(2023, 5, 15)
        encoder = JSONEncoder()
        result = encoder.encode({"date": d})
        expected = '{"date": "2023-05-15"}'
        self.assertEqual(json.loads(result), json.loads(expected))

    def test_encode_decimal(self):
        """Test encoding Decimal objects"""
        dec = Decimal("123.45")
        encoder = JSONEncoder()
        result = encoder.encode({"amount": dec})
        expected = '{"amount": 123.45}'
        self.assertEqual(json.loads(result), json.loads(expected))

    def test_encode_enum(self):
        """Test encoding Enum objects"""
        enum_val = TestEnum.OPTION1
        encoder = JSONEncoder()
        result = encoder.encode({"option": enum_val})
        expected = '{"option": "option1"}'
        self.assertEqual(json.loads(result), json.loads(expected))

    def test_encode_bytes(self):
        """Test encoding bytes objects"""
        b = b"test bytes"
        encoder = JSONEncoder()
        result = encoder.encode({"data": b})
        # Bytes should be base64 encoded
        self.assertIn("data", json.loads(result))
        # The exact encoding can vary, so just check that we get a string back
        self.assertIsInstance(json.loads(result)["data"], str)

    def test_encode_unsupported_type(self):
        """Test encoding an unsupported type raises TypeError"""
        class UnsupportedType:
            pass

        obj = UnsupportedType()
        encoder = JSONEncoder()
        with self.assertRaises(TypeError):
            encoder.encode({"obj": obj})


class TestJSONSerialization(unittest.TestCase):
    """Test JSON serialization and deserialization"""

    def test_serialize_deserialize_complex_object(self):
        """Test serializing and deserializing a complex object"""
        original = {
            "name": "Trading Strategy 1",
            "created_at": datetime(2023, 5, 15, 12, 30, 45),
            "active": True,
            "parameters": {
                "risk_level": TestEnum.OPTION2,
                "max_position": Decimal("10000.50"),
                "min_profit": 0.05,
            },
            "assets": ["BTC", "ETH", "SOL"],
            "binary_data": b"some binary data"
        }

        # Serialize to JSON
        json_str = serialize_to_json(original)
        self.assertIsInstance(json_str, str)

        # Deserialize from JSON
        deserialized = deserialize_from_json(json_str)

        # Check that complex types were properly reconstructed
        self.assertIsInstance(deserialized["created_at"], datetime)
        self.assertEqual(deserialized["created_at"].year, 2023)
        self.assertEqual(deserialized["created_at"].month, 5)
        self.assertEqual(deserialized["created_at"].day, 15)

        # Enum should be reconstructed as a string
        self.assertEqual(deserialized["parameters"]["risk_level"], "option2")

        # Decimal should be reconstructed as a float
        self.assertIsInstance(deserialized["parameters"]["max_position"], float)
        self.assertAlmostEqual(deserialized["parameters"]["max_position"], 10000.50)

        # Bytes should be properly reconstructed
        self.assertIsInstance(deserialized["binary_data"], bytes)

    def test_serialize_to_json_error(self):
        """Test error handling in serialize_to_json"""
        # Create an object that cannot be serialized to JSON
        class UnserializableObject:
            def __repr__(self):
                return "UnserializableObject()"

        obj = {"problematic": UnserializableObject()}

        with self.assertRaises(SerializationError):
            serialize_to_json(obj)

    def test_deserialize_from_json_error(self):
        """Test error handling in deserialize_from_json"""
        invalid_json = "{invalid: json"

        with self.assertRaises(DeserializationError):
            deserialize_from_json(invalid_json)


@pytest.mark.skipif(True, reason="Requires bson module")
class TestBSONSerialization(unittest.TestCase):
    """Test BSON serialization and deserialization"""

    @patch('utils.serializers.bson')
    def test_serialize_to_bson(self, mock_bson):
        """Test serializing to BSON"""
        data = {"name": "Test", "value": 123}
        serialize_to_bson(data)
        mock_bson.encode.assert_called_once()

    @patch('utils.serializers.bson')
    def test_deserialize_from_bson(self, mock_bson):
        """Test deserializing from BSON"""
        bson_data = b'some bson data'
        mock_bson.decode.return_value = {"name": "Test", "value": 123}
        result = deserialize_from_bson(bson_data)
        self.assertEqual(result, {"name": "Test", "value": 123})
        mock_bson.decode.assert_called_once_with(bson_data)

    @patch('utils.serializers.bson')
    def test_serialize_to_bson_error(self, mock_bson):
        """Test error handling in serialize_to_bson"""
        mock_bson.encode.side_effect = Exception("BSON encoding error")
        with self.assertRaises(SerializationError):
            serialize_to_bson({"test": "data"})

    @patch('utils.serializers.bson')
    def test_deserialize_from_bson_error(self, mock_bson):
        """Test error handling in deserialize_from_bson"""
        mock_bson.decode.side_effect = Exception("BSON decoding error")
        with self.assertRaises(DeserializationError):
            deserialize_from_bson(b'invalid bson')


@pytest.mark.skipif(True, reason="Requires msgpack module")
class TestMessagePackSerialization(unittest.TestCase):
    """Test MessagePack serialization and deserialization"""

    @patch('utils.serializers.msgpack')
    def test_serialize_to_msgpack(self, mock_msgpack):
        """Test serializing to MessagePack"""
        data = {"name": "Test", "value": 123}
        serialize_to_msgpack(data)
        mock_msgpack.packb.assert_called_once()

    @patch('utils.serializers.msgpack')
    def test_deserialize_from_msgpack(self, mock_msgpack):
        """Test deserializing from MessagePack"""
        msgpack_data = b'some msgpack data'
        mock_msgpack.unpackb.return_value = {"name": "Test", "value": 123}
        result = deserialize_from_msgpack(msgpack_data)
        self.assertEqual(result, {"name": "Test", "value": 123})
        mock_msgpack.unpackb.assert_called_once()

    @patch('utils.serializers.msgpack')
    def test_serialize_to_msgpack_error(self, mock_msgpack):
        """Test error handling in serialize_to_msgpack"""
        mock_msgpack.packb.side_effect = Exception("MessagePack encoding error")
        with self.assertRaises(SerializationError):
            serialize_to_msgpack({"test": "data"})

    @patch('utils.serializers.msgpack')
    def test_deserialize_from_msgpack_error(self, mock_msgpack):
        """Test error handling in deserialize_from_msgpack"""
        mock_msgpack.unpackb.side_effect = Exception("MessagePack decoding error")
        with self.assertRaises(DeserializationError):
            deserialize_from_msgpack(b'invalid msgpack')


@pytest.mark.skipif(True, reason="Requires avro module")
class TestAvroSerialization(unittest.TestCase):
    """Test Avro serialization and deserialization"""

    def setUp(self):
        """Set up test schema"""
        self.schema = {
            "namespace": "trading.avro",
            "type": "record",
            "name": "Order",
            "fields": [
                {"name": "symbol", "type": "string"},
                {"name": "quantity", "type": "double"},
                {"name": "price", "type": "double"},
                {"name": "side", "type": "string"}
            ]
        }

    @patch('utils.serializers.avro')
    def test_serialize_to_avro(self, mock_avro):
        """Test serializing to Avro"""
        data = {
            "symbol": "BTC/USD",
            "quantity": 1.5,
            "price": 50000.0,
            "side": "buy"
        }
        serialize_to_avro(data, self.schema)
        mock_avro.io.DatumWriter.assert_called_once()
        mock_avro.schema.Parse.assert_called_once()

    @patch('utils.serializers.avro')
    def test_deserialize_from_avro(self, mock_avro):
        """Test deserializing from Avro"""
        avro_data = b'some avro data'
        mock_reader = Mock()
        mock_avro.io.DatumReader.return_value = mock_reader
        mock_reader.read.return_value = {
            "symbol": "BTC/USD",
            "quantity": 1.5,
            "price": 50000.0,
            "side": "buy"
        }

        result = deserialize_from_avro(avro_data, self.schema)

        self.assertEqual(result["symbol"], "BTC/USD")
        self.assertEqual(result["quantity"], 1.5)
        self.assertEqual(result["price"], 50000.0)
        self.assertEqual(result["side"], "buy")

        mock_avro.io.DatumReader.assert_called_once()
        mock_avro.schema.Parse.assert_called_once()

    @patch('utils.serializers.avro')
    def test_serialize_to_avro_error(self, mock_avro):
        """Test error handling in serialize_to_avro"""
        mock_avro.io.DatumWriter.side_effect = Exception("Avro encoding error")
        with self.assertRaises(SerializationError):
            serialize_to_avro({"test": "data"}, self.schema)

    @patch('utils.serializers.avro')
    def test_deserialize_from_avro_error(self, mock_avro):
        """Test error handling in deserialize_from_avro"""
        mock_avro.io.DatumReader.side_effect = Exception("Avro decoding error")
        with self.assertRaises(DeserializationError):
            deserialize_from_avro(b'invalid avro', self.schema)


@pytest.mark.skipif(True, reason="Requires protobuf module")
class TestProtobufSerialization(unittest.TestCase):
    """Test Protobuf serialization and deserialization"""

    @patch('utils.serializers.get_protobuf_class')
    def test_serialize_to_protobuf(self, mock_get_protobuf_class):
        """Test serializing to Protobuf"""
        mock_proto_class = Mock()
        mock_get_protobuf_class.return_value = mock_proto_class

        data = {"name": "Test", "value": 123}
        proto_instance = Mock()
        mock_proto_class.return_value = proto_instance
        proto_instance.SerializeToString.return_value = b'serialized protobuf'

        result = serialize_to_protobuf(data, "TestMessage")

        self.assertEqual(result, b'serialized protobuf')
        mock_get_protobuf_class.assert_called_once_with("TestMessage")
        mock_proto_class.assert_called_once()
        proto_instance.SerializeToString.assert_called_once()

    @patch('utils.serializers.get_protobuf_class')
    def test_deserialize_from_protobuf(self, mock_get_protobuf_class):
        """Test deserializing from Protobuf"""
        mock_proto_class = Mock()
        mock_get_protobuf_class.return_value = mock_proto_class

        proto_data = b'some protobuf data'
        proto_instance = Mock()
        mock_proto_class.return_value = proto_instance
        proto_instance.ParseFromString.return_value = None

        # Mock the conversion from protobuf to dict
        mock_proto_to_dict = Mock()
        mock_proto_to_dict.return_value = {"name": "Test", "value": 123}

        with patch('utils.serializers.protobuf_to_dict', mock_proto_to_dict):
            result = deserialize_from_protobuf(proto_data, "TestMessage")

        self.assertEqual(result, {"name": "Test", "value": 123})
        mock_get_protobuf_class.assert_called_once_with("TestMessage")
        mock_proto_class.assert_called_once()
        proto_instance.ParseFromString.assert_called_once_with(proto_data)
        mock_proto_to_dict.assert_called_once_with(proto_instance)

    @patch('utils.serializers.get_protobuf_class')
    def test_serialize_to_protobuf_error(self, mock_get_protobuf_class):
        """Test error handling in serialize_to_protobuf"""
        mock_proto_class = Mock()
        mock_get_protobuf_class.return_value = mock_proto_class

        proto_instance = Mock()
        mock_proto_class.return_value = proto_instance
        proto_instance.SerializeToString.side_effect = Exception("Protobuf encoding error")

        with self.assertRaises(SerializationError):
            serialize_to_protobuf({"test": "data"}, "TestMessage")

    @patch('utils.serializers.get_protobuf_class')
    def test_deserialize_from_protobuf_error(self, mock_get_protobuf_class):
        """Test error handling in deserialize_from_protobuf"""
        mock_proto_class = Mock()
        mock_get_protobuf_class.return_value = mock_proto_class

        proto_instance = Mock()
        mock_proto_class.return_value = proto_instance
        proto_instance.ParseFromString.side_effect = Exception("Protobuf decoding error")

        with self.assertRaises(DeserializationError):
            deserialize_from_protobuf(b'invalid protobuf', "TestMessage")


if __name__ == '__main__':
    unittest.main()