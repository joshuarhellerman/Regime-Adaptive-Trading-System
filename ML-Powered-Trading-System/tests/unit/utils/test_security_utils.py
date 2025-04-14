"""
Test cases for security_utils.py module.
"""

import base64
import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest import mock

import jwt
import pytest
from cryptography.fernet import Fernet

from utils.security_utils import (SecurityUtils, KeyStoreType, encrypt_data,
                                 decrypt_data, generate_password_hash,
                                 verify_password, generate_api_key_pair)


class TestSecurityUtils:
    """Test cases for SecurityUtils class."""

    @pytest.fixture
    def security_utils(self, tmp_path):
        """Create a SecurityUtils instance with a temporary key store path."""
        key_store_path = tmp_path / "keys"
        key_store_path.mkdir(exist_ok=True)
        config = {
            'key_store_type': 'file',
            'key_store_path': str(key_store_path),
        }
        return SecurityUtils(config)

    def test_initialization(self, security_utils):
        """Test initialization of SecurityUtils."""
        assert security_utils.key_store_type == KeyStoreType.FILE
        assert isinstance(security_utils.encryption_key, bytes)
        assert isinstance(security_utils.jwt_secret, bytes)
        assert security_utils.jwt_algorithm == 'HS256'
        assert security_utils.jwt_expiry == 3600

    def test_encryption_decryption(self, security_utils):
        """Test encryption and decryption of data."""
        # Test with string
        plaintext = "Secret message"
        encrypted = security_utils.encrypt(plaintext)
        decrypted = security_utils.decrypt(encrypted)
        assert decrypted == plaintext

        # Test with dict
        data_dict = {"key": "value", "nested": {"id": 123}}
        encrypted = security_utils.encrypt(data_dict)
        decrypted = security_utils.decrypt(encrypted, 'json')
        assert decrypted == data_dict

        # Test with bytes
        data_bytes = b"Binary data"
        encrypted = security_utils.encrypt(data_bytes)
        decrypted = security_utils.decrypt(encrypted, 'bytes')
        assert decrypted == data_bytes

    def test_hash_generation_verification(self, security_utils):
        """Test hash generation and verification."""
        data = "Data to hash"
        hash_value = security_utils.generate_hash(data)
        assert security_utils.verify_hash(data, hash_value)
        assert not security_utils.verify_hash("Different data", hash_value)

        # Test with different algorithm
        hash_value_md5 = security_utils.generate_hash(data, 'md5')
        assert security_utils.verify_hash(data, hash_value_md5, 'md5')
        assert hash_value != hash_value_md5

    def test_hmac_generation_verification(self, security_utils):
        """Test HMAC generation and verification."""
        data = "Data to sign"
        key = "secret_key"
        signature = security_utils.generate_hmac(data, key)
        assert security_utils.verify_hmac(data, signature, key)
        assert not security_utils.verify_hmac("Different data", signature, key)
        assert not security_utils.verify_hmac(data, signature, "wrong_key")

    def test_jwt_token_creation_verification(self, security_utils):
        """Test JWT token creation and verification."""
        payload = {"user_id": 123, "role": "admin"}
        token = security_utils.create_jwt_token(payload)
        verified_payload = security_utils.verify_jwt_token(token)
        assert verified_payload["user_id"] == payload["user_id"]
        assert verified_payload["role"] == payload["role"]

        # Test token with issuer and audience
        token = security_utils.create_jwt_token(
            payload, issuer="test_issuer", audience="test_audience"
        )
        verified_payload = security_utils.verify_jwt_token(
            token, issuer="test_issuer", audience="test_audience"
        )
        assert verified_payload["iss"] == "test_issuer"
        assert verified_payload["aud"] == "test_audience"

        # Test token expiration
        with mock.patch('utils.security_utils.datetime') as mock_datetime:
            # Create a token that expires in 10 seconds
            mock_datetime.utcnow.return_value = datetime.utcnow()
            token = security_utils.create_jwt_token(payload, expiry=10)

            # Fast forward time by 11 seconds
            mock_datetime.utcnow.return_value = datetime.utcnow() + timedelta(seconds=11)

            # Verify token should raise ExpiredSignatureError
            with pytest.raises(jwt.ExpiredSignatureError):
                security_utils.verify_jwt_token(token)

    def test_password_handling(self, security_utils):
        """Test password hashing and verification."""
        password = "SecureP@ssw0rd"
        hashed = security_utils.generate_password_hash(password)
        assert security_utils.verify_password(password, hashed)
        assert not security_utils.verify_password("WrongPassword", hashed)

    def test_key_storage_retrieval(self, security_utils, tmp_path):
        """Test key storage and retrieval."""
        key_name = "test_key"
        test_key = b"secret_test_key_data"

        # Store key
        assert security_utils._store_key(key_name, test_key)

        # Load key
        loaded_key = security_utils._load_key(key_name)
        assert loaded_key == test_key

    def test_env_key_storage(self):
        """Test key storage and retrieval using environment variables."""
        # Create SecurityUtils with ENV key store
        config = {
            'key_store_type': 'env',
            'key_env_prefix': 'TEST_KEY_'
        }
        security_utils = SecurityUtils(config)

        key_name = "test_env_key"
        test_key = b"secret_test_env_key_data"

        # Store key
        assert security_utils._store_key(key_name, test_key)

        # Verify env var was set
        env_var = f"TEST_KEY_{key_name.upper()}"
        assert env_var in os.environ
        assert base64.urlsafe_b64decode(os.environ[env_var]) == test_key

        # Load key
        loaded_key = security_utils._load_key(key_name)
        assert loaded_key == test_key

        # Clean up
        del os.environ[env_var]

    def test_api_key_generation(self, security_utils):
        """Test API key generation."""
        api_key = security_utils.generate_api_key()
        assert isinstance(api_key, str)
        assert len(api_key) > 0

        # Test API key pair
        api_key, api_secret = security_utils.generate_api_key_pair()
        assert isinstance(api_key, str)
        assert isinstance(api_secret, str)
        assert api_key != api_secret

    def test_secure_api_credentials(self, security_utils):
        """Test API credential storage and retrieval."""
        credentials = {
            "provider": "test_provider",
            "api_key": "test_api_key",
            "api_secret": "test_api_secret"
        }

        # Store credentials
        metadata = security_utils.secure_api_credentials(credentials)
        assert "credential_id" in metadata
        assert metadata["provider"] == "test_provider"

        # Retrieve credentials
        retrieved = security_utils.retrieve_api_credentials(metadata["credential_id"])
        assert retrieved == credentials

        # Rotate key
        updated = security_utils.rotate_api_key(metadata["credential_id"])
        assert updated["api_key"] != credentials["api_key"]
        assert updated["api_secret"] != credentials["api_secret"]
        assert "rotated_at" in updated

    def test_rsa_key_operations(self, security_utils):
        """Test RSA key pair generation and operations."""
        # Generate key pair
        private_key, public_key = security_utils.generate_rsa_key_pair(1024)  # Smaller key for faster tests
        assert private_key.startswith(b"-----BEGIN PRIVATE KEY-----")
        assert public_key.startswith(b"-----BEGIN PUBLIC KEY-----")

        # Test encryption and decryption
        data = "Secret RSA message"
        encrypted = security_utils.encrypt_with_rsa(data, public_key)
        decrypted = security_utils.decrypt_with_rsa(encrypted, private_key)
        assert decrypted.decode('utf-8') == data

        # Test signing and verification
        signature = security_utils.sign_with_rsa(data, private_key)
        assert security_utils.verify_rsa_signature(data, signature, public_key)
        assert not security_utils.verify_rsa_signature("Different data", signature, public_key)

    def test_derive_key_from_password(self, security_utils):
        """Test key derivation from password."""
        password = "test_password"
        salt = b"test_salt"

        key1 = security_utils._derive_key_from_password(password, salt)
        key2 = security_utils._derive_key_from_password(password, salt)
        assert key1 == key2  # Same password and salt should give same key

        key3 = security_utils._derive_key_from_password("different_password", salt)
        assert key1 != key3  # Different password should give different key

        key4 = security_utils._derive_key_from_password(password, b"different_salt")
        assert key1 != key4  # Different salt should give different key

    def test_totp_generation_verification(self, security_utils):
        """Test TOTP generation and verification."""
        # Generate a TOTP secret
        secret = security_utils.generate_otp_secret()
        assert isinstance(secret, str)
        assert len(secret) > 0

        # Generate and verify TOTP
        totp = security_utils.generate_totp(secret)
        assert len(totp) == 6  # Default is 6 digits
        assert security_utils.verify_totp(secret, totp)

    def test_request_signing(self, security_utils):
        """Test API request signing."""
        # Test generic signing
        method = "POST"
        url = "https://api.example.com/v1/order"
        params = {"symbol": "BTC-USD"}
        data = {"amount": 1.0, "price": 50000}
        api_key = "test_key"
        api_secret = "test_secret"

        headers = security_utils.sign_request(
            method, url, params, data, api_key, api_secret
        )

        assert "headers" in headers
        assert headers["headers"]["API-Key"] == api_key
        assert "API-Timestamp" in headers["headers"]
        assert "API-Signature" in headers["headers"]

        # Test Binance signing
        binance_headers = security_utils.sign_request(
            method, url, params, data, api_key, api_secret, exchange="binance"
        )

        assert "headers" in binance_headers
        assert binance_headers["headers"]["X-MBX-APIKEY"] == api_key
        assert "params" in binance_headers
        assert "timestamp" in binance_headers["params"]
        assert "signature" in binance_headers["params"]

    def test_secure_random_string(self, security_utils):
        """Test secure random string generation."""
        # Default settings
        random_str = security_utils.generate_secure_random_string()
        assert len(random_str) == 16
        assert any(c.isupper() for c in random_str)
        assert any(c.islower() for c in random_str)
        assert any(c.isdigit() for c in random_str)
        assert not any(c in "!@#$%^&*()-_=+[]{}|;:,.<>?" for c in random_str)

        # Custom settings
        random_str = security_utils.generate_secure_random_string(
            length=20,
            include_upper=False,
            include_lower=True,
            include_digits=False,
            include_special=True
        )
        assert len(random_str) == 20
        assert not any(c.isupper() for c in random_str)
        assert any(c.islower() for c in random_str)
        assert not any(c.isdigit() for c in random_str)
        assert any(c in "!@#$%^&*()-_=+[]{}|;:,.<>?" for c in random_str)

    def test_secure_filename(self, security_utils):
        """Test secure filename generation."""
        # Test normal filename
        filename = "test_file.txt"
        secure_name = security_utils.generate_secure_filename(filename)
        assert secure_name == filename

        # Test with path components
        filename = "/path/to/test_file.txt"
        secure_name = security_utils.generate_secure_filename(filename)
        assert secure_name == "test_file.txt"

        # Test with dangerous characters
        filename = "malicious;<script>.php"
        secure_name = security_utils.generate_secure_filename(filename)
        assert secure_name == "malicious.php"

        # Test empty filename
        filename = ""
        secure_name = security_utils.generate_secure_filename(filename)
        assert secure_name == "file"


class TestModuleFunctions:
    """Test cases for module-level utility functions."""

    def test_encrypt_decrypt_data(self, tmp_path):
        """Test module-level encryption/decryption functions."""
        # Configure a key store path for testing
        key_store_path = tmp_path / "keys"
        key_store_path.mkdir(exist_ok=True)
        config = {
            'key_store_type': 'file',
            'key_store_path': str(key_store_path),
        }

        # Test with string
        plaintext = "Secret message"
        encrypted = encrypt_data(plaintext, config)
        decrypted = decrypt_data(encrypted, config=config)
        assert decrypted == plaintext

        # Test with dict
        data_dict = {"key": "value", "nested": {"id": 123}}
        encrypted = encrypt_data(data_dict, config)
        decrypted = decrypt_data(encrypted, 'json', config)
        assert decrypted == data_dict

    def test_password_hash_verify(self, tmp_path):
        """Test module-level password hash functions."""
        # Configure a key store path for testing
        key_store_path = tmp_path / "keys"
        key_store_path.mkdir(exist_ok=True)
        config = {
            'key_store_type': 'file',
            'key_store_path': str(key_store_path),
        }

        password = "SecureP@ssw0rd"
        hashed = generate_password_hash(password, config)
        assert verify_password(password, hashed, config)
        assert not verify_password("WrongPassword", hashed, config)

    def test_api_key_generation_module(self, tmp_path):
        """Test module-level API key generation."""
        # Configure a key store path for testing
        key_store_path = tmp_path / "keys"
        key_store_path.mkdir(exist_ok=True)
        config = {
            'key_store_type': 'file',
            'key_store_path': str(key_store_path),
        }

        api_key, api_secret = generate_api_key_pair(config)
        assert isinstance(api_key, str)
        assert isinstance(api_secret, str)
        assert api_key != api_secret

    @mock.patch.dict(os.environ, {"TRADING_ENCRYPTION_KEY": base64.urlsafe_b64encode(Fernet.generate_key()).decode()})
    def test_fallback_key_from_env(self):
        """Test fallback to environment variable for encryption key."""
        # Create SecurityUtils without valid key store
        config = {
            'key_store_type': 'file',
            'key_store_path': '/non/existent/path',
        }

        security_utils = SecurityUtils(config)

        # Should still be able to encrypt/decrypt using env fallback
        plaintext = "Secret message"
        encrypted = security_utils.encrypt(plaintext)
        decrypted = security_utils.decrypt(encrypted)
        assert decrypted == plaintext