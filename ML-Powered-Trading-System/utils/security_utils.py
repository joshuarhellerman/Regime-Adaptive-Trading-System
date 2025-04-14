"""
utils/security_utils.py - Security Utilities

This module provides security-related utilities for the trading system, including:
- Secure credential storage and management
- Encryption/decryption utilities
- Hash generation and verification
- Token creation and validation
- Certificate handling
- Password management
"""

import base64
import hashlib
import hmac
import json
import logging
import os
import secrets
import time
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import jwt
from cryptography.fernet import Fernet
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.x509 import load_pem_x509_certificate

from utils.logger import get_logger


class KeyStoreType(Enum):
    """Types of key storage"""
    FILE = "file"
    ENV = "env"
    KEYRING = "keyring"
    VAULT = "vault"
    AWS_KMS = "aws_kms"
    AZURE_KEYVAULT = "azure_keyvault"
    CUSTOM = "custom"


class SecurityUtils:
    """
    Security utilities for the trading system.

    This class provides methods for:
    - Secure credential storage
    - Encryption and decryption
    - Hash generation and verification
    - JWT token management
    - API request signing
    - Certificate handling
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize security utilities.

        Args:
            config: Configuration for security utils
        """
        self.logger = get_logger(__name__)
        self.config = config or {}

        # Key storage settings
        self.key_store_type = KeyStoreType(self.config.get('key_store_type', 'file'))
        self.key_store_path = Path(self.config.get('key_store_path', 'config/keys'))
        self.key_env_prefix = self.config.get('key_env_prefix', 'TRADING_KEY_')

        # Encryption settings
        self.encryption_key = self._load_or_create_encryption_key()
        self.fernet = Fernet(self.encryption_key)

        # JWT settings
        self.jwt_secret = self._load_or_create_jwt_secret()
        self.jwt_algorithm = self.config.get('jwt_algorithm', 'HS256')
        self.jwt_expiry = self.config.get('jwt_expiry', 3600)  # 1 hour default

        # Hash settings
        self.default_hash_algorithm = self.config.get('default_hash_algorithm', 'sha256')

        self.logger.info("Security utilities initialized")

    def _load_or_create_encryption_key(self) -> bytes:
        """
        Load or create a Fernet encryption key.

        Returns:
            bytes: The encryption key
        """
        key_name = 'encryption_key'

        try:
            # Try to load existing key
            key = self._load_key(key_name)
            if key:
                return key

            # Generate new key if not found
            self.logger.info("Generating new encryption key")
            key = Fernet.generate_key()
            self._store_key(key_name, key)
            return key

        except Exception as e:
            self.logger.error(f"Error managing encryption key: {str(e)}")
            # Fallback to environment-based key or generate temporary one
            return self._generate_fallback_key("TRADING_ENCRYPTION_KEY")

    def _load_or_create_jwt_secret(self) -> bytes:
        """
        Load or create a JWT secret key.

        Returns:
            bytes: The JWT secret key
        """
        key_name = 'jwt_secret'

        try:
            # Try to load existing key
            key = self._load_key(key_name)
            if key:
                return key

            # Generate new key if not found
            self.logger.info("Generating new JWT secret")
            key = secrets.token_bytes(32)
            self._store_key(key_name, key)
            return key

        except Exception as e:
            self.logger.error(f"Error managing JWT secret: {str(e)}")
            # Fallback to environment-based key or generate temporary one
            return self._generate_fallback_key("TRADING_JWT_SECRET")

    def _generate_fallback_key(self, env_var: str) -> bytes:
        """
        Generate a fallback key from environment or create temporary one.

        Args:
            env_var: Environment variable name for the key

        Returns:
            bytes: The fallback key
        """
        # Check environment variable
        if env_var in os.environ:
            key_str = os.environ[env_var]
            try:
                # Try to decode as base64
                return base64.urlsafe_b64decode(key_str)
            except:
                # Use as seed to generate key
                return self._derive_key_from_password(key_str, b"fallback_salt")

        # Generate temporary key with warning
        self.logger.warning(f"Using temporary key (will change at restart). Set {env_var} for persistence.")
        return Fernet.generate_key()

    def _load_key(self, key_name: str) -> Optional[bytes]:
        """
        Load a key from the configured key store.

        Args:
            key_name: Name of the key

        Returns:
            bytes or None: The key if found, None otherwise
        """
        if self.key_store_type == KeyStoreType.FILE:
            return self._load_key_from_file(key_name)
        elif self.key_store_type == KeyStoreType.ENV:
            return self._load_key_from_env(key_name)
        elif self.key_store_type == KeyStoreType.KEYRING:
            return self._load_key_from_keyring(key_name)
        elif self.key_store_type == KeyStoreType.VAULT:
            return self._load_key_from_vault(key_name)
        elif self.key_store_type == KeyStoreType.AWS_KMS:
            return self._load_key_from_aws_kms(key_name)
        elif self.key_store_type == KeyStoreType.AZURE_KEYVAULT:
            return self._load_key_from_azure_keyvault(key_name)
        else:
            self.logger.warning(f"Unsupported key store type: {self.key_store_type}")
            return None

    def _store_key(self, key_name: str, key: bytes) -> bool:
        """
        Store a key in the configured key store.

        Args:
            key_name: Name of the key
            key: The key to store

        Returns:
            bool: True if successful, False otherwise
        """
        if self.key_store_type == KeyStoreType.FILE:
            return self._store_key_to_file(key_name, key)
        elif self.key_store_type == KeyStoreType.ENV:
            return self._store_key_to_env(key_name, key)
        elif self.key_store_type == KeyStoreType.KEYRING:
            return self._store_key_to_keyring(key_name, key)
        elif self.key_store_type == KeyStoreType.VAULT:
            return self._store_key_to_vault(key_name, key)
        elif self.key_store_type == KeyStoreType.AWS_KMS:
            return self._store_key_to_aws_kms(key_name, key)
        elif self.key_store_type == KeyStoreType.AZURE_KEYVAULT:
            return self._store_key_to_azure_keyvault(key_name, key)
        else:
            self.logger.warning(f"Unsupported key store type: {self.key_store_type}")
            return False

    def _load_key_from_file(self, key_name: str) -> Optional[bytes]:
        """
        Load a key from a file.

        Args:
            key_name: Name of the key

        Returns:
            bytes or None: The key if found, None otherwise
        """
        key_path = self.key_store_path / f"{key_name}.key"
        if not key_path.exists():
            return None

        try:
            with open(key_path, 'rb') as f:
                return base64.urlsafe_b64decode(f.read())
        except Exception as e:
            self.logger.error(f"Error loading key {key_name} from file: {str(e)}")
            return None

    def _store_key_to_file(self, key_name: str, key: bytes) -> bool:
        """
        Store a key to a file.

        Args:
            key_name: Name of the key
            key: The key to store

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Create directory if it doesn't exist
            self.key_store_path.mkdir(parents=True, exist_ok=True)

            # Set secure permissions (700)
            os.chmod(self.key_store_path, 0o700)

            # Write key to file
            key_path = self.key_store_path / f"{key_name}.key"
            with open(key_path, 'wb') as f:
                f.write(base64.urlsafe_b64encode(key))

            # Set secure permissions (600)
            os.chmod(key_path, 0o600)

            return True
        except Exception as e:
            self.logger.error(f"Error storing key {key_name} to file: {str(e)}")
            return False

    def _load_key_from_env(self, key_name: str) -> Optional[bytes]:
        """
        Load a key from environment variable.

        Args:
            key_name: Name of the key

        Returns:
            bytes or None: The key if found, None otherwise
        """
        env_var = f"{self.key_env_prefix}{key_name.upper()}"
        if env_var not in os.environ:
            return None

        try:
            return base64.urlsafe_b64decode(os.environ[env_var])
        except Exception as e:
            self.logger.error(f"Error loading key {key_name} from environment: {str(e)}")
            return None

    def _store_key_to_env(self, key_name: str, key: bytes) -> bool:
        """
        Store a key to environment variable.

        Args:
            key_name: Name of the key
            key: The key to store

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            env_var = f"{self.key_env_prefix}{key_name.upper()}"
            os.environ[env_var] = base64.urlsafe_b64encode(key).decode('utf-8')
            return True
        except Exception as e:
            self.logger.error(f"Error storing key {key_name} to environment: {str(e)}")
            return False

    def _load_key_from_keyring(self, key_name: str) -> Optional[bytes]:
        """
        Load a key from system keyring.

        Args:
            key_name: Name of the key

        Returns:
            bytes or None: The key if found, None otherwise
        """
        try:
            import keyring
            value = keyring.get_password("trading_system", key_name)
            if value:
                return base64.urlsafe_b64decode(value)
            return None
        except ImportError:
            self.logger.error("Keyring module not available")
            return None
        except Exception as e:
            self.logger.error(f"Error loading key {key_name} from keyring: {str(e)}")
            return None

    def _store_key_to_keyring(self, key_name: str, key: bytes) -> bool:
        """
        Store a key to system keyring.

        Args:
            key_name: Name of the key
            key: The key to store

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            import keyring
            keyring.set_password("trading_system", key_name, base64.urlsafe_b64encode(key).decode('utf-8'))
            return True
        except ImportError:
            self.logger.error("Keyring module not available")
            return False
        except Exception as e:
            self.logger.error(f"Error storing key {key_name} to keyring: {str(e)}")
            return False

    def _load_key_from_vault(self, key_name: str) -> Optional[bytes]:
        """
        Load a key from HashiCorp Vault.

        Args:
            key_name: Name of the key

        Returns:
            bytes or None: The key if found, None otherwise
        """
        # Placeholder for Vault integration
        self.logger.warning("Vault integration not implemented yet")
        return None

    def _store_key_to_vault(self, key_name: str, key: bytes) -> bool:
        """
        Store a key to HashiCorp Vault.

        Args:
            key_name: Name of the key
            key: The key to store

        Returns:
            bool: True if successful, False otherwise
        """
        # Placeholder for Vault integration
        self.logger.warning("Vault integration not implemented yet")
        return False

    def _load_key_from_aws_kms(self, key_name: str) -> Optional[bytes]:
        """
        Load a key from AWS KMS.

        Args:
            key_name: Name of the key

        Returns:
            bytes or None: The key if found, None otherwise
        """
        # Placeholder for AWS KMS integration
        self.logger.warning("AWS KMS integration not implemented yet")
        return None

    def _store_key_to_aws_kms(self, key_name: str, key: bytes) -> bool:
        """
        Store a key to AWS KMS.

        Args:
            key_name: Name of the key
            key: The key to store

        Returns:
            bool: True if successful, False otherwise
        """
        # Placeholder for AWS KMS integration
        self.logger.warning("AWS KMS integration not implemented yet")
        return False

    def _load_key_from_azure_keyvault(self, key_name: str) -> Optional[bytes]:
        """
        Load a key from Azure KeyVault.

        Args:
            key_name: Name of the key

        Returns:
            bytes or None: The key if found, None otherwise
        """
        # Placeholder for Azure KeyVault integration
        self.logger.warning("Azure KeyVault integration not implemented yet")
        return None

    def _store_key_to_azure_keyvault(self, key_name: str, key: bytes) -> bool:
        """
        Store a key to Azure KeyVault.

        Args:
            key_name: Name of the key
            key: The key to store

        Returns:
            bool: True if successful, False otherwise
        """
        # Placeholder for Azure KeyVault integration
        self.logger.warning("Azure KeyVault integration not implemented yet")
        return False

    def encrypt(self, data: Union[str, bytes, Dict, List]) -> str:
        """
        Encrypt data using the system encryption key.

        Args:
            data: Data to encrypt (string, bytes, dict, or list)

        Returns:
            str: Base64-encoded encrypted data
        """
        try:
            # Convert data to bytes
            if isinstance(data, str):
                data_bytes = data.encode('utf-8')
            elif isinstance(data, (dict, list)):
                data_bytes = json.dumps(data).encode('utf-8')
            elif isinstance(data, bytes):
                data_bytes = data
            else:
                raise ValueError(f"Unsupported data type: {type(data)}")

            # Encrypt data
            encrypted = self.fernet.encrypt(data_bytes)

            # Return as base64 string
            return base64.urlsafe_b64encode(encrypted).decode('utf-8')

        except Exception as e:
            self.logger.error(f"Encryption error: {str(e)}")
            raise

    def decrypt(self, encrypted_data: str, output_type: str = 'str') -> Union[str, bytes, Dict, List]:
        """
        Decrypt data using the system encryption key.

        Args:
            encrypted_data: Base64-encoded encrypted data
            output_type: Type of output ('str', 'bytes', 'json')

        Returns:
            The decrypted data in the requested format
        """
        try:
            # Decode base64
            encrypted_bytes = base64.urlsafe_b64decode(encrypted_data)

            # Decrypt data
            decrypted_bytes = self.fernet.decrypt(encrypted_bytes)

            # Return in the requested format
            if output_type == 'bytes':
                return decrypted_bytes
            elif output_type == 'json':
                return json.loads(decrypted_bytes.decode('utf-8'))
            else:
                return decrypted_bytes.decode('utf-8')

        except Exception as e:
            self.logger.error(f"Decryption error: {str(e)}")
            raise

    def generate_hash(self, data: Union[str, bytes], algorithm: str = None) -> str:
        """
        Generate a hash of the data.

        Args:
            data: Data to hash
            algorithm: Hash algorithm (default: from config)

        Returns:
            str: Hex-encoded hash
        """
        algorithm = algorithm or self.default_hash_algorithm

        try:
            # Convert data to bytes
            if isinstance(data, str):
                data_bytes = data.encode('utf-8')
            elif isinstance(data, bytes):
                data_bytes = data
            else:
                raise ValueError(f"Unsupported data type: {type(data)}")

            # Create hash
            h = hashlib.new(algorithm)
            h.update(data_bytes)

            # Return hex digest
            return h.hexdigest()

        except Exception as e:
            self.logger.error(f"Hash generation error: {str(e)}")
            raise

    def verify_hash(self, data: Union[str, bytes], hash_value: str, algorithm: str = None) -> bool:
        """
        Verify that a hash matches the data.

        Args:
            data: Data to verify
            hash_value: Expected hash
            algorithm: Hash algorithm (default: from config)

        Returns:
            bool: True if hash matches, False otherwise
        """
        try:
            calculated_hash = self.generate_hash(data, algorithm)
            return hmac.compare_digest(calculated_hash, hash_value)
        except Exception as e:
            self.logger.error(f"Hash verification error: {str(e)}")
            return False

    def generate_hmac(self, data: Union[str, bytes], key: Union[str, bytes], algorithm: str = None) -> str:
        """
        Generate an HMAC signature.

        Args:
            data: Data to sign
            key: Key for signing
            algorithm: HMAC algorithm (default: from config)

        Returns:
            str: Hex-encoded HMAC
        """
        algorithm = algorithm or self.default_hash_algorithm

        try:
            # Convert data and key to bytes
            if isinstance(data, str):
                data_bytes = data.encode('utf-8')
            elif isinstance(data, bytes):
                data_bytes = data
            else:
                raise ValueError(f"Unsupported data type: {type(data)}")

            if isinstance(key, str):
                key_bytes = key.encode('utf-8')
            elif isinstance(key, bytes):
                key_bytes = key
            else:
                raise ValueError(f"Unsupported key type: {type(key)}")

            # Create HMAC
            h = hmac.new(key_bytes, data_bytes, getattr(hashlib, algorithm))

            # Return hex digest
            return h.hexdigest()

        except Exception as e:
            self.logger.error(f"HMAC generation error: {str(e)}")
            raise

    def verify_hmac(self, data: Union[str, bytes], signature: str, key: Union[str, bytes], algorithm: str = None) -> bool:
        """
        Verify an HMAC signature.

        Args:
            data: Data to verify
            signature: Expected HMAC signature
            key: Key for verification
            algorithm: HMAC algorithm (default: from config)

        Returns:
            bool: True if signature matches, False otherwise
        """
        try:
            calculated_hmac = self.generate_hmac(data, key, algorithm)
            return hmac.compare_digest(calculated_hmac, signature)
        except Exception as e:
            self.logger.error(f"HMAC verification error: {str(e)}")
            return False

    def sign_request(self, method: str, url: str, params: Dict = None, data: Dict = None,
                     api_key: str = None, api_secret: str = None, exchange: str = None) -> Dict[str, str]:
        """
        Generate authentication headers for an API request.

        Args:
            method: HTTP method
            url: Request URL
            params: URL parameters
            data: Request body data
            api_key: API key
            api_secret: API secret
            exchange: Exchange name (for exchange-specific signing)

        Returns:
            Dict: Headers for authenticated request
        """
        if exchange == "binance":
            return self._sign_binance_request(method, url, params, data, api_key, api_secret)
        elif exchange == "coinbase":
            return self._sign_coinbase_request(method, url, params, data, api_key, api_secret)
        elif exchange == "ftx":
            return self._sign_ftx_request(method, url, params, data, api_key, api_secret)
        elif exchange == "kraken":
            return self._sign_kraken_request(method, url, params, data, api_key, api_secret)
        else:
            # Generic signing (timestamp + signature)
            return self._sign_generic_request(method, url, params, data, api_key, api_secret)

    def _sign_binance_request(self, method: str, url: str, params: Dict = None, data: Dict = None,
                              api_key: str = None, api_secret: str = None) -> Dict[str, str]:
        """
        Sign a request for Binance API.

        Args:
            method: HTTP method
            url: Request URL
            params: URL parameters
            data: Request body data
            api_key: API key
            api_secret: API secret

        Returns:
            Dict: Headers for authenticated request
        """
        # Combine parameters and data
        request_params = {}
        if params:
            request_params.update(params)
        if data:
            request_params.update(data)

        # Add timestamp
        request_params['timestamp'] = int(time.time() * 1000)

        # Create query string
        query_string = '&'.join([f"{k}={v}" for k, v in sorted(request_params.items())])

        # Create signature
        signature = self.generate_hmac(query_string, api_secret, 'sha256')

        # Add signature to parameters
        request_params['signature'] = signature

        # Return headers and updated parameters
        return {
            'headers': {
                'X-MBX-APIKEY': api_key
            },
            'params': request_params
        }

    def _sign_coinbase_request(self, method: str, url: str, params: Dict = None, data: Dict = None,
                               api_key: str = None, api_secret: str = None) -> Dict[str, str]:
        """
        Sign a request for Coinbase API.

        Args:
            method: HTTP method
            url: Request URL
            params: URL parameters
            data: Request body data
            api_key: API key
            api_secret: API secret

        Returns:
            Dict: Headers for authenticated request
        """
        # Get timestamp
        timestamp = str(int(time.time()))

        # Create prehash string
        path = url.split('coinbase.com')[-1]
        body = json.dumps(data) if data else ''
        prehash_string = timestamp + method.upper() + path + body

        # Create signature
        signature = self.generate_hmac(prehash_string, base64.b64decode(api_secret), 'sha256')
        signature_b64 = base64.b64encode(bytes.fromhex(signature)).decode('utf-8')

        # Return headers
        return {
            'headers': {
                'CB-ACCESS-KEY': api_key,
                'CB-ACCESS-SIGN': signature_b64,
                'CB-ACCESS-TIMESTAMP': timestamp,
                'Content-Type': 'application/json'
            }
        }

    def _sign_ftx_request(self, method: str, url: str, params: Dict = None, data: Dict = None,
                          api_key: str = None, api_secret: str = None) -> Dict[str, str]:
        """
        Sign a request for FTX API.

        Args:
            method: HTTP method
            url: Request URL
            params: URL parameters
            data: Request body data
            api_key: API key
            api_secret: API secret

        Returns:
            Dict: Headers for authenticated request
        """
        # Get timestamp
        ts = int(time.time() * 1000)

        # Create signature payload
        path = url.split('ftx.com')[-1]
        if params:
            path += '?' + '&'.join([f"{k}={v}" for k, v in sorted(params.items())])

        payload = f'{ts}{method.upper()}{path}'
        if data:
            payload += json.dumps(data)

        # Create signature
        signature = self.generate_hmac(payload, api_secret, 'sha256')

        # Return headers
        return {
            'headers': {
                'FTX-KEY': api_key,
                'FTX-SIGN': signature,
                'FTX-TS': str(ts)
            }
        }

    def _sign_kraken_request(self, method: str, url: str, params: Dict = None, data: Dict = None,
                             api_key: str = None, api_secret: str = None) -> Dict[str, str]:
        """
        Sign a request for Kraken API.

        Args:
            method: HTTP method
            url: Request URL
            params: URL parameters
            data: Request body data
            api_key: API key
            api_secret: API secret

        Returns:
            Dict: Headers for authenticated request
        """
        # Placeholder for Kraken-specific signing
        self.logger.warning("Kraken signing not fully implemented yet")

        # Return generic headers
        return self._sign_generic_request(method, url, params, data, api_key, api_secret)

    def _sign_generic_request(self, method: str, url: str, params: Dict = None, data: Dict = None,
                              api_key: str = None, api_secret: str = None) -> Dict[str, str]:
        """
        Sign a request using a generic approach.

        Args:
            method: HTTP method
            url: Request URL
            params: URL parameters
            data: Request body data
            api_key: API key
            api_secret: API secret

        Returns:
            Dict: Headers for authenticated request
        """
        # Get timestamp
        timestamp = str(int(time.time() * 1000))

        # Create signature data
        signature_data = {
            'method': method.upper(),
            'url': url,
            'timestamp': timestamp
        }

        if params:
            signature_data['params'] = params
        if data:
            signature_data['data'] = data

        # Create signature
        signature_str = json.dumps(signature_data, sort_keys=True)
        signature = self.generate_hmac(signature_str, api_secret, 'sha256')

        # Return headers
        return {
            'headers': {
                'API-Key': api_key,
                'API-Timestamp': timestamp,
                'API-Signature': signature
            }
        }

    def create_jwt_token(self, payload: Dict[str, Any], expiry: int = None,
                        issuer: str = None, audience: str = None) -> str:
        """
        Create a JWT token.

        Args:
            payload: Token payload
            expiry: Expiry time in seconds (default: from config)
            issuer: Token issuer
            audience: Token audience

        Returns:
            str: JWT token
        """
        try:
            # Set expiry if not provided
            expiry = expiry or self.jwt_expiry

            # Create complete payload
            full_payload = payload.copy()

            # Add standard claims
            current_time = datetime.utcnow()
            full_payload.update({
                'iat': current_time,
                'exp': current_time + timedelta(seconds=expiry),
                'nbf': current_time
            })

            # Add optional claims
            if issuer:
                full_payload['iss'] = issuer
            if audience:
                full_payload['aud'] = audience

            # Create JWT token
            token = jwt.encode(
                full_payload,
                self.jwt_secret,
                algorithm=self.jwt_algorithm
            )

            return token

        except Exception as e:
            self.logger.error(f"JWT creation error: {str(e)}")
            raise

    def verify_jwt_token(self, token: str, issuer: str = None, audience: str = None) -> Dict[str, Any]:
        """
        Verify a JWT token.

        Args:
            token: JWT token
            issuer: Expected issuer
            audience: Expected audience

        Returns:
            Dict: Token payload if valid

        Raises:
            jwt.InvalidTokenError: If token is invalid
        """
        try:
            # Set verification options
            options = {
                'verify_signature': True,
                'verify_exp': True,
                'verify_nbf': True,
                'verify_iat': True
            }

            # Add issuer and audience verification if provided
            if issuer:
                options['verify_iss'] = True
            if audience:
                options['verify_aud'] = True

            # Verify token
            payload = jwt.decode(
                token,
                self.jwt_secret,
                algorithms=[self.jwt_algorithm],
                issuer=issuer,
                audience=audience,
                options=options
            )

            return payload

        except jwt.ExpiredSignatureError:
            self.logger.error("JWT token has expired")
            raise
        except jwt.InvalidTokenError as e:
            self.logger.error(f"JWT verification error: {str(e)}")
            raise

    def derive_key_from_password(self, password: str, salt: Optional[bytes] = None) -> bytes:
        """
        Derive a cryptographic key from a password.

        Args:
            password: Password
            salt: Salt (optional)

        Returns:
            bytes: Derived key
        """
        # Create or use salt
        if salt is None:
            salt = os.urandom(16)

        try:
            # Create key derivation function
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
                backend=default_backend()
            )

            # Derive key
            key = kdf.derive(password.encode('utf-8'))

            return key

        except Exception as e:
            self.logger.error(f"Key derivation error: {str(e)}")
            raise

    def _derive_key_from_password(self, password: str, salt: bytes) -> bytes:
        """
        Internal method to derive a key from a password.

        Args:
            password: Password
            salt: Salt

        Returns:
            bytes: Derived key
        """
        try:
            # Create key derivation function
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
                backend=default_backend()
            )

            # Derive key
            key = base64.urlsafe_b64encode(
                kdf.derive(password.encode('utf-8'))
            )

            return key

        except Exception as e:
            self.logger.error(f"Key derivation error: {str(e)}")
            raise

    def generate_password_hash(self, password: str) -> str:
        """
        Generate a secure hash of a password.

        Args:
            password: Password to hash

        Returns:
            str: Hashed password
        """
        try:
            # Generate salt
            salt = os.urandom(16)

            # Derive key from password
            key = self.derive_key_from_password(password, salt)

            # Combine salt and key
            combined = salt + key

            # Encode as base64
            encoded = base64.urlsafe_b64encode(combined).decode('utf-8')

            return encoded

        except Exception as e:
            self.logger.error(f"Password hash generation error: {str(e)}")
            raise

    def verify_password(self, password: str, password_hash: str) -> bool:
        """
        Verify a password against a hash.

        Args:
            password: Password to verify
            password_hash: Hashed password

        Returns:
            bool: True if password matches hash, False otherwise
        """
        try:
            # Decode hash
            decoded = base64.urlsafe_b64decode(password_hash)

            # Extract salt and hash
            salt = decoded[:16]
            stored_key = decoded[16:]

            # Derive key from password
            key = self.derive_key_from_password(password, salt)

            # Compare keys
            return hmac.compare_digest(key, stored_key)

        except Exception as e:
            self.logger.error(f"Password verification error: {str(e)}")
            return False

    def generate_otp_secret(self) -> str:
        """
        Generate a secret for TOTP (Time-based One-Time Password).

        Returns:
            str: Base32-encoded secret
        """
        try:
            # Generate random bytes
            secret_bytes = secrets.token_bytes(20)

            # Encode as base32
            secret = base64.b32encode(secret_bytes).decode('utf-8')

            return secret

        except Exception as e:
            self.logger.error(f"OTP secret generation error: {str(e)}")
            raise

    def generate_totp(self, secret: str, digits: int = 6, period: int = 30) -> str:
        """
        Generate a TOTP code.

        Args:
            secret: Base32-encoded secret
            digits: Number of digits in the code
            period: Time period in seconds

        Returns:
            str: TOTP code
        """
        try:
            # Import pyotp if available
            try:
                import pyotp
                totp = pyotp.TOTP(secret, digits=digits, interval=period)
                return totp.now()
            except ImportError:
                # Fallback implementation
                return self._generate_totp_fallback(secret, digits, period)

        except Exception as e:
            self.logger.error(f"TOTP generation error: {str(e)}")
            raise

    def _generate_totp_fallback(self, secret: str, digits: int = 6, period: int = 30) -> str:
        """
        Fallback implementation of TOTP.

        Args:
            secret: Base32-encoded secret
            digits: Number of digits in the code
            period: Time period in seconds

        Returns:
            str: TOTP code
        """
        try:
            # Decode secret
            secret_bytes = base64.b32decode(secret)

            # Get current time
            counter = int(time.time() // period)

            # Convert counter to bytes
            counter_bytes = counter.to_bytes(8, 'big')

            # Generate HMAC
            hmac_hash = hmac.new(secret_bytes, counter_bytes, 'sha1').digest()

            # Dynamic truncation
            offset = hmac_hash[-1] & 0x0F
            truncated_hash = (
                ((hmac_hash[offset] & 0x7F) << 24) |
                ((hmac_hash[offset + 1] & 0xFF) << 16) |
                ((hmac_hash[offset + 2] & 0xFF) << 8) |
                (hmac_hash[offset + 3] & 0xFF)
            )

            # Generate code
            code = truncated_hash % (10 ** digits)

            # Pad with leading zeros
            return str(code).zfill(digits)

        except Exception as e:
            self.logger.error(f"TOTP fallback generation error: {str(e)}")
            raise

    def verify_totp(self, secret: str, code: str, digits: int = 6, period: int = 30, window: int = 1) -> bool:
        """
        Verify a TOTP code.

        Args:
            secret: Base32-encoded secret
            code: TOTP code to verify
            digits: Number of digits in the code
            period: Time period in seconds
            window: Time window for validation (in periods)

        Returns:
            bool: True if code is valid, False otherwise
        """
        try:
            # Import pyotp if available
            try:
                import pyotp
                totp = pyotp.TOTP(secret, digits=digits, interval=period)
                return totp.verify(code, valid_window=window)
            except ImportError:
                # Fallback implementation
                current_time = int(time.time())
                for i in range(-window, window + 1):
                    time_offset = current_time + (i * period)
                    counter = int(time_offset // period)
                    generated_code = self._generate_totp_for_counter(secret, counter, digits)
                    if hmac.compare_digest(generated_code, code):
                        return True
                return False

        except Exception as e:
            self.logger.error(f"TOTP verification error: {str(e)}")
            return False

    def _generate_totp_for_counter(self, secret: str, counter: int, digits: int = 6) -> str:
        """
        Generate TOTP for a specific counter value.

        Args:
            secret: Base32-encoded secret
            counter: Counter value
            digits: Number of digits in the code

        Returns:
            str: TOTP code
        """
        try:
            # Decode secret
            secret_bytes = base64.b32decode(secret)

            # Convert counter to bytes
            counter_bytes = counter.to_bytes(8, 'big')

            # Generate HMAC
            hmac_hash = hmac.new(secret_bytes, counter_bytes, 'sha1').digest()

            # Dynamic truncation
            offset = hmac_hash[-1] & 0x0F
            truncated_hash = (
                ((hmac_hash[offset] & 0x7F) << 24) |
                ((hmac_hash[offset + 1] & 0xFF) << 16) |
                ((hmac_hash[offset + 2] & 0xFF) << 8) |
                (hmac_hash[offset + 3] & 0xFF)
            )

            # Generate code
            code = truncated_hash % (10 ** digits)

            # Pad with leading zeros
            return str(code).zfill(digits)

        except Exception as e:
            self.logger.error(f"TOTP counter generation error: {str(e)}")
            raise

    def generate_api_key(self, length: int = 32) -> str:
        """
        Generate a random API key.

        Args:
            length: Length of the key in bytes

        Returns:
            str: API key
        """
        try:
            # Generate random bytes
            random_bytes = secrets.token_bytes(length)

            # Encode as base64
            encoded = base64.urlsafe_b64encode(random_bytes).decode('utf-8')

            # Remove padding
            encoded = encoded.rstrip('=')

            return encoded

        except Exception as e:
            self.logger.error(f"API key generation error: {str(e)}")
            raise

    def generate_api_key_pair(self) -> Tuple[str, str]:
        """
        Generate an API key and secret pair.

        Returns:
            Tuple: (api_key, api_secret)
        """
        try:
            # Generate API key
            api_key = self.generate_api_key(24)

            # Generate API secret
            api_secret = self.generate_api_key(32)

            return api_key, api_secret

        except Exception as e:
            self.logger.error(f"API key pair generation error: {str(e)}")
            raise

    def secure_api_credentials(self, credentials: Dict[str, str]) -> Dict[str, str]:
        """
        Securely store API credentials.

        Args:
            credentials: API credentials

        Returns:
            Dict: Metadata for stored credentials
        """
        try:
            # Generate a unique ID for the credentials
            credential_id = secrets.token_hex(8)

            # Encrypt credentials
            encrypted = self.encrypt(credentials)

            # Store encrypted credentials
            self._store_key(f"api_cred_{credential_id}", encrypted.encode('utf-8'))

            # Return metadata
            return {
                'credential_id': credential_id,
                'created_at': datetime.now().isoformat(),
                'provider': credentials.get('provider', 'unknown')
            }

        except Exception as e:
            self.logger.error(f"API credential storage error: {str(e)}")
            raise

    def retrieve_api_credentials(self, credential_id: str) -> Dict[str, str]:
        """
        Retrieve stored API credentials.

        Args:
            credential_id: Credential ID

        Returns:
            Dict: API credentials
        """
        try:
            # Retrieve encrypted credentials
            encrypted = self._load_key(f"api_cred_{credential_id}")

            if not encrypted:
                raise ValueError(f"Credentials not found for ID: {credential_id}")

            # Decrypt credentials
            credentials = self.decrypt(encrypted.decode('utf-8'), 'json')

            return credentials

        except Exception as e:
            self.logger.error(f"API credential retrieval error: {str(e)}")
            raise

    def rotate_api_key(self, credential_id: str) -> Dict[str, str]:
        """
        Rotate an API key pair.

        Args:
            credential_id: Credential ID

        Returns:
            Dict: New API credentials
        """
        try:
            # Retrieve existing credentials
            credentials = self.retrieve_api_credentials(credential_id)

            # Generate new key pair
            api_key, api_secret = self.generate_api_key_pair()

            # Update credentials
            credentials['api_key'] = api_key
            credentials['api_secret'] = api_secret
            credentials['rotated_at'] = datetime.now().isoformat()

            # Store updated credentials
            self._store_key(f"api_cred_{credential_id}", self.encrypt(credentials).encode('utf-8'))

            return credentials

        except Exception as e:
            self.logger.error(f"API key rotation error: {str(e)}")
            raise

    def generate_rsa_key_pair(self, key_size: int = 2048) -> Tuple[bytes, bytes]:
        """
        Generate an RSA key pair.

        Args:
            key_size: Key size in bits

        Returns:
            Tuple: (private_key, public_key)
        """
        try:
            # Generate private key
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=key_size,
                backend=default_backend()
            )

            # Get public key
            public_key = private_key.public_key()

            # Serialize private key
            private_pem = private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            )

            # Serialize public key
            public_pem = public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            )

            return private_pem, public_pem

        except Exception as e:
            self.logger.error(f"RSA key pair generation error: {str(e)}")
            raise

    def encrypt_with_rsa(self, data: Union[str, bytes], public_key: bytes) -> bytes:
        """
        Encrypt data with an RSA public key.

        Args:
            data: Data to encrypt
            public_key: RSA public key

        Returns:
            bytes: Encrypted data
        """
        try:
            # Convert data to bytes
            if isinstance(data, str):
                data_bytes = data.encode('utf-8')
            elif isinstance(data, bytes):
                data_bytes = data
            else:
                raise ValueError(f"Unsupported data type: {type(data)}")

            # Load public key
            key = serialization.load_pem_public_key(
                public_key,
                backend=default_backend()
            )

            # Encrypt data
            encrypted = key.encrypt(
                data_bytes,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )

            return encrypted

        except Exception as e:
            self.logger.error(f"RSA encryption error: {str(e)}")
            raise

    def decrypt_with_rsa(self, encrypted_data: bytes, private_key: bytes) -> bytes:
        """
        Decrypt data with an RSA private key.

        Args:
            encrypted_data: Encrypted data
            private_key: RSA private key

        Returns:
            bytes: Decrypted data
        """
        try:
            # Load private key
            key = serialization.load_pem_private_key(
                private_key,
                password=None,
                backend=default_backend()
            )

            # Decrypt data
            decrypted = key.decrypt(
                encrypted_data,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )

            return decrypted

        except Exception as e:
            self.logger.error(f"RSA decryption error: {str(e)}")
            raise

    def sign_with_rsa(self, data: Union[str, bytes], private_key: bytes) -> bytes:
        """
        Sign data with an RSA private key.

        Args:
            data: Data to sign
            private_key: RSA private key

        Returns:
            bytes: Signature
        """
        try:
            # Convert data to bytes
            if isinstance(data, str):
                data_bytes = data.encode('utf-8')
            elif isinstance(data, bytes):
                data_bytes = data
            else:
                raise ValueError(f"Unsupported data type: {type(data)}")

            # Load private key
            key = serialization.load_pem_private_key(
                private_key,
                password=None,
                backend=default_backend()
            )

            # Sign data
            signature = key.sign(
                data_bytes,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )

            return signature

        except Exception as e:
            self.logger.error(f"RSA signing error: {str(e)}")
            raise

    def verify_rsa_signature(self, data: Union[str, bytes], signature: bytes, public_key: bytes) -> bool:
        """
        Verify an RSA signature.

        Args:
            data: Data to verify
            signature: Signature to verify
            public_key: RSA public key

        Returns:
            bool: True if signature is valid, False otherwise
        """
        try:
            # Convert data to bytes
            if isinstance(data, str):
                data_bytes = data.encode('utf-8')
            elif isinstance(data, bytes):
                data_bytes = data
            else:
                raise ValueError(f"Unsupported data type: {type(data)}")

            # Load public key
            key = serialization.load_pem_public_key(
                public_key,
                backend=default_backend()
            )

            # Verify signature
            key.verify(
                signature,
                data_bytes,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )

            return True

        except Exception as e:
            self.logger.error(f"RSA signature verification error: {str(e)}")
            return False

    def verify_certificate(self, certificate: bytes, trusted_certs: List[bytes] = None) -> bool:
        """
        Verify a certificate.

        Args:
            certificate: Certificate to verify
            trusted_certs: List of trusted certificates

        Returns:
            bool: True if certificate is valid, False otherwise
        """
        try:
            # Load certificate
            cert = load_pem_x509_certificate(certificate, default_backend())

            # Basic validation - check if expired
            now = datetime.utcnow()
            if now < cert.not_valid_before or now > cert.not_valid_after:
                self.logger.warning("Certificate is expired or not yet valid")
                return False

            # If no trusted certificates provided, just check validity
            if not trusted_certs:
                return True

            # TODO: Implement certificate chain validation
            self.logger.warning("Certificate chain validation not implemented yet")
            return True

        except Exception as e:
            self.logger.error(f"Certificate verification error: {str(e)}")
            return False

    def generate_secure_filename(self, filename: str) -> str:
        """
        Generate a secure version of a filename.

        Args:
            filename: Original filename

        Returns:
            str: Secure filename
        """
        try:
            # Remove path components
            filename = os.path.basename(filename)

            # Replace potentially dangerous characters
            secure_name = ''.join(c for c in filename if c.isalnum() or c in '._- ')

            # Ensure filename is not empty
            if not secure_name:
                secure_name = 'file'

            return secure_name

        except Exception as e:
            self.logger.error(f"Secure filename generation error: {str(e)}")
            return 'file'

    def generate_secure_random_string(self, length: int = 16,
                                     include_upper: bool = True,
                                     include_lower: bool = True,
                                     include_digits: bool = True,
                                     include_special: bool = False) -> str:
        """
        Generate a secure random string.

        Args:
            length: Length of the string
            include_upper: Include uppercase letters
            include_lower: Include lowercase letters
            include_digits: Include digits
            include_special: Include special characters

        Returns:
            str: Random string
        """
        try:
            # Define character sets
            chars = ''
            if include_upper:
                chars += 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
            if include_lower:
                chars += 'abcdefghijklmnopqrstuvwxyz'
            if include_digits:
                chars += '0123456789'
            if include_special:
                chars += '!@#$%^&*()-_=+[]{}|;:,.<>?'

            # Ensure at least one character set is selected
            if not chars:
                chars = 'abcdefghijklmnopqrstuvwxyz'

            # Generate random string
            return ''.join(secrets.choice(chars) for _ in range(length))

        except Exception as e:
            self.logger.error(f"Secure random string generation error: {str(e)}")
            raise

    def sanitize_input(self, input_str: str) -> str:
        """
        Sanitize user input to prevent injection attacks.

        Args:
            input_str: Input string

        Returns:
            str: Sanitized string
        """
        try:
            # Basic sanitization
            sanitized = input_str.replace('<', '&lt;').replace('>', '&gt;')

            return sanitized

        except Exception as e:
            self.logger.error(f"Input sanitization error: {str(e)}")
            raise


# Module-level utility functions for easy access

def encrypt_data(data: Union[str, bytes, Dict, List], config: Dict[str, Any] = None) -> str:
    """
    Encrypt data using the system encryption.

    Args:
        data: Data to encrypt
        config: Security configuration (optional)

    Returns:
        str: Encrypted data
    """
    security_utils = SecurityUtils(config)
    return security_utils.encrypt(data)

def decrypt_data(encrypted_data: str, output_type: str = 'str', config: Dict[str, Any] = None) -> Union[str, bytes, Dict, List]:
    """
    Decrypt data using the system encryption.

    Args:
        encrypted_data: Encrypted data
        output_type: Output type ('str', 'bytes', 'json')
        config: Security configuration (optional)

    Returns:
        The decrypted data
    """
    security_utils = SecurityUtils(config)
    return security_utils.decrypt(encrypted_data, output_type)

def generate_password_hash(password: str, config: Dict[str, Any] = None) -> str:
    """
    Generate a secure hash of a password.

    Args:
        password: Password to hash
        config: Security configuration (optional)

    Returns:
        str: Hashed password
    """
    security_utils = SecurityUtils(config)
    return security_utils.generate_password_hash(password)

def verify_password(password: str, password_hash: str, config: Dict[str, Any] = None) -> bool:
    """
    Verify a password against a hash.

    Args:
        password: Password to verify
        password_hash: Hashed password
        config: Security configuration (optional)

    Returns:
        bool: True if password matches hash
    """
    security_utils = SecurityUtils(config)
    return security_utils.verify_password(password, password_hash)

def generate_api_key_pair(config: Dict[str, Any] = None) -> Tuple[str, str]:
    """
    Generate an API key and secret pair.

    Args:
        config: Security configuration (optional)

    Returns:
        Tuple: (api_key, api_secret)
    """
    security_utils = SecurityUtils(config)
    return security_utils.generate_api_key_pair()

def sign_request(method: str, url: str, params: Dict = None, data: Dict = None,
                api_key: str = None, api_secret: str = None, exchange: str = None,
                config: Dict[str, Any] = None) -> Dict[str, str]:
    """
    Generate authentication headers for an API request.

    Args:
        method: HTTP method
        url: Request URL
        params: URL parameters
        data: Request body data
        api_key: API key
        api_secret: API secret
        exchange: Exchange name
        config: Security configuration (optional)

    Returns:
        Dict: Headers for authenticated request
    """
    security_utils = SecurityUtils(config)
    return security_utils.sign_request(method, url, params, data, api_key, api_secret, exchange)