"""
Exchange configuration module that defines settings for exchange connections
and API credentials.

This module provides configuration for connecting to various exchanges,
managing API credentials, and setting exchange-specific parameters.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Union
from pathlib import Path
import os

from .base_config import BaseConfig, ConfigManager


class ExchangeType(Enum):
    """Enumeration of exchange types."""
    SPOT = "spot"
    FUTURES = "futures"
    OPTIONS = "options"
    MARGIN = "margin"


class AssetClass(Enum):
    """Enumeration of asset classes."""
    CRYPTO = "crypto"
    FOREX = "forex"
    EQUITIES = "equities"
    COMMODITIES = "commodities"
    FIXED_INCOME = "fixed_income"
    INDICES = "indices"


class ExchangeProvider(Enum):
    """Enumeration of exchange providers."""
    BINANCE = "binance"
    COINBASE = "coinbase"
    FTX = "ftx"
    OANDA = "oanda"
    INTERACTIVE_BROKERS = "interactive_brokers"
    ALPACA = "alpaca"
    KRAKEN = "kraken"
    BITSTAMP = "bitstamp"
    GEMINI = "gemini"
    CUSTOM = "custom"


@dataclass
class RateLimitConfig(BaseConfig):
    """Configuration for exchange rate limits."""
    # Rate limit settings
    requests_per_second: float = 10.0
    requests_per_minute: int = 600
    requests_per_hour: int = 30000

    # Weight-based limits (some exchanges use request weights)
    weight_per_second: float = 50.0
    weight_per_minute: int = 3000
    weight_per_hour: int = 100000

    # Burst settings
    burst_factor: float = 2.0  # Allow bursts up to this factor of normal rate

    # Behavior when approaching limits
    throttle_at_percent: float = 80.0  # Start throttling when this percent of limit is reached

    def validate(self) -> List[str]:
        """Validate the rate limit configuration."""
        errors = super().validate()

        # Validate rates
        if self.requests_per_second < 0:
            errors.append("requests_per_second cannot be negative")

        if self.requests_per_minute < 0:
            errors.append("requests_per_minute cannot be negative")

        if self.requests_per_hour < 0:
            errors.append("requests_per_hour cannot be negative")

        # Validate weights
        if self.weight_per_second < 0:
            errors.append("weight_per_second cannot be negative")

        if self.weight_per_minute < 0:
            errors.append("weight_per_minute cannot be negative")

        if self.weight_per_hour < 0:
            errors.append("weight_per_hour cannot be negative")

        # Validate burst factor
        if self.burst_factor < 1:
            errors.append("burst_factor must be at least 1")

        # Validate throttle percentage
        if not (0 <= self.throttle_at_percent <= 100):
            errors.append("throttle_at_percent must be between 0 and 100")

        return errors


@dataclass
class APICredentials(BaseConfig):
    """Configuration for API credentials."""
    # API keys
    api_key: str = ""
    api_secret: str = ""
    passphrase: str = ""  # Some exchanges require a passphrase

    # Additional authentication
    token: str = ""
    refresh_token: str = ""

    # Certificate-based authentication
    certificate_path: str = ""
    private_key_path: str = ""

    # API permissions
    read_only: bool = False
    trading_enabled: bool = True
    withdrawal_enabled: bool = False

    def validate(self) -> List[str]:
        """Validate the API credentials."""
        errors = super().validate()

        # Basic validation of required credentials
        if not self.api_key:
            errors.append("api_key must not be empty")

        if not self.api_secret:
            errors.append("api_secret must not be empty")

        # Validate certificate paths if specified
        if self.certificate_path and not os.path.exists(self.certificate_path):
            errors.append(f"Certificate file not found: {self.certificate_path}")

        if self.private_key_path and not os.path.exists(self.private_key_path):
            errors.append(f"Private key file not found: {self.private_key_path}")

        return errors


@dataclass
class ExchangeInstanceConfig(BaseConfig):
    """Configuration for a single exchange instance."""
    # Exchange identification
    instance_id: str = ""
    provider: ExchangeProvider = ExchangeProvider.CUSTOM
    name: str = ""
    description: str = ""

    # Exchange type and asset classes
    exchange_type: ExchangeType = ExchangeType.SPOT
    asset_classes: List[AssetClass] = field(default_factory=list)

    # Connection settings
    base_url: str = ""
    websocket_url: str = ""
    use_testnet: bool = False

    # Timeout settings
    connection_timeout_ms: int = 5000
    read_timeout_ms: int = 10000

    # API credentials
    credentials: APICredentials = field(default_factory=APICredentials)

    # Rate limits
    rate_limits: RateLimitConfig = field(default_factory=RateLimitConfig)

    # Proxy settings
    use_proxy: bool = False
    proxy_url: str = ""

    # Retry settings
    retry_attempts: int = 3
    retry_delay_ms: int = 1000

    # Trading settings
    trading_enabled: bool = True
    default_fee_rate: float = 0.001  # 0.1% default fee
    maker_fee_rate: float = 0.0009  # 0.09% maker fee
    taker_fee_rate: float = 0.0018  # 0.18% taker fee

    # Exchange-specific parameters
    parameters: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize the exchange instance configuration."""
        super().__post_init__()

        # Convert provider from string if needed
        if isinstance(self.provider, str):
            try:
                self.provider = ExchangeProvider(self.provider.lower())
            except ValueError:
                print(f"Warning: Invalid provider '{self.provider}', defaulting to CUSTOM")
                self.provider = ExchangeProvider.CUSTOM

        # Convert exchange_type from string if needed
        if isinstance(self.exchange_type, str):
            try:
                self.exchange_type = ExchangeType(self.exchange_type.lower())
            except ValueError:
                print(f"Warning: Invalid exchange_type '{self.exchange_type}', defaulting to SPOT")
                self.exchange_type = ExchangeType.SPOT

        # Convert asset_classes from strings if needed
        asset_class_list = []
        for asset_class in self.asset_classes:
            if isinstance(asset_class, str):
                try:
                    asset_class = AssetClass(asset_class.lower())
                except ValueError:
                    print(f"Warning: Invalid asset_class '{asset_class}', skipping")
                    continue
            asset_class_list.append(asset_class)
        self.asset_classes = asset_class_list

        # Set default URLs based on provider if not specified
        if not self.base_url:
            if self.provider == ExchangeProvider.BINANCE:
                self.base_url = "https://api.binance.com" if not self.use_testnet else "https://testnet.binance.vision"
            elif self.provider == ExchangeProvider.COINBASE:
                self.base_url = "https://api.coinbase.com"
            elif self.provider == ExchangeProvider.FTX:
                self.base_url = "https://ftx.com/api" if not self.use_testnet else "https://ftx.us/api"
            elif self.provider == ExchangeProvider.OANDA:
                self.base_url = "https://api-fxtrade.oanda.com" if not self.use_testnet else "https://api-fxpractice.oanda.com"

        if not self.websocket_url:
            if self.provider == ExchangeProvider.BINANCE:
                self.websocket_url = "wss://stream.binance.com:9443/ws" if not self.use_testnet else "wss://testnet.binance.vision/ws"
            elif self.provider == ExchangeProvider.COINBASE:
                self.websocket_url = "wss://ws-feed.pro.coinbase.com"
            elif self.provider == ExchangeProvider.FTX:
                self.websocket_url = "wss://ftx.com/ws" if not self.use_testnet else "wss://ftx.us/ws"
            elif self.provider == ExchangeProvider.OANDA:
                self.websocket_url = "wss://stream-fxtrade.oanda.com" if not self.use_testnet else "wss://stream-fxpractice.oanda.com"

        # Set name based on provider if not specified
        if not self.name:
            self.name = self.provider.value.capitalize()

        # Ensure credentials is an APICredentials object
        if isinstance(self.credentials, dict):
            self.credentials = APICredentials(**self.credentials)

        # Ensure rate_limits is a RateLimitConfig object
        if isinstance(self.rate_limits, dict):
            self.rate_limits = RateLimitConfig(**self.rate_limits)

        # Set appropriate rate limits based on the provider
        if self.provider == ExchangeProvider.BINANCE:
            self.rate_limits.requests_per_minute = 1200
            self.rate_limits.weight_per_minute = 6000
        elif self.provider == ExchangeProvider.COINBASE:
            self.rate_limits.requests_per_second = 5
        elif self.provider == ExchangeProvider.FTX:
            self.rate_limits.requests_per_second = 30

    def validate(self) -> List[str]:
        """Validate the exchange instance configuration."""
        errors = super().validate()

        # Validate instance_id
        if not self.instance_id:
            errors.append("instance_id must not be empty")

        # Validate URLs
        if not self.base_url:
            errors.append("base_url must not be empty")

        if not self.websocket_url:
            errors.append("websocket_url must not be empty")

        # Validate timeouts
        if self.connection_timeout_ms <= 0:
            errors.append("connection_timeout_ms must be positive")

        if self.read_timeout_ms <= 0:
            errors.append("read_timeout_ms must be positive")

        # Validate proxy settings
        if self.use_proxy and not self.proxy_url:
            errors.append("proxy_url must not be empty when use_proxy is enabled")

        # Validate retry settings
        if self.retry_attempts < 0:
            errors.append("retry_attempts cannot be negative")

        if self.retry_delay_ms <= 0:
            errors.append("retry_delay_ms must be positive")

        # Validate fee rates
        if self.default_fee_rate < 0:
            errors.append("default_fee_rate cannot be negative")

        if self.maker_fee_rate < 0:
            errors.append("maker_fee_rate cannot be negative")

        if self.taker_fee_rate < 0:
            errors.append("taker_fee_rate cannot be negative")

        # Validate asset classes
        if not self.asset_classes:
            errors.append("At least one asset_class must be specified")

        # Validate credentials
        credential_errors = self.credentials.validate()
        for error in credential_errors:
            errors.append(f"In credentials: {error}")

        # Validate rate limits
        rate_limit_errors = self.rate_limits.validate()
        for error in rate_limit_errors:
            errors.append(f"In rate_limits: {error}")

        return errors


@dataclass
class WebhookConfig(BaseConfig):
    """Configuration for exchange webhooks."""
    # Webhook settings
    enable_webhooks: bool = False
    webhook_url: str = ""
    webhook_secret: str = ""

    # Events to subscribe to
    subscribe_to_trades: bool = True
    subscribe_to_orders: bool = True
    subscribe_to_account: bool = False

    # Security settings
    verify_signatures: bool = True

    def validate(self) -> List[str]:
        """Validate the webhook configuration."""
        errors = super().validate()

        # Only validate if webhooks are enabled
        if not self.enable_webhooks:
            return errors

        # Validate webhook URL
        if not self.webhook_url:
            errors.append("webhook_url must not be empty when enable_webhooks is enabled")

        # Validate webhook secret
        if self.verify_signatures and not self.webhook_secret:
            errors.append("webhook_secret must not be empty when verify_signatures is enabled")

        return errors


@dataclass
class OrderRoutingConfig(BaseConfig):
    """Configuration for order routing."""
    # Routing settings
    enable_smart_routing: bool = True
    primary_exchange: str = ""
    backup_exchanges: List[str] = field(default_factory=list)

    # Routing criteria
    route_by_liquidity: bool = True
    route_by_fees: bool = True
    route_by_latency: bool = True

    # Fallback settings
    max_routing_attempts: int = 3
    routing_timeout_ms: int = 2000

    def validate(self) -> List[str]:
        """Validate the order routing configuration."""
        errors = super().validate()

        # Only validate if smart routing is enabled
        if not self.enable_smart_routing:
            return errors

        # Validate primary exchange
        if not self.primary_exchange:
            errors.append("primary_exchange must not be empty when enable_smart_routing is enabled")

        # Validate routing attempts
        if self.max_routing_attempts <= 0:
            errors.append("max_routing_attempts must be positive")

        # Validate routing timeout
        if self.routing_timeout_ms <= 0:
            errors.append("routing_timeout_ms must be positive")

        return errors


@dataclass
class MarketDataConfig(BaseConfig):
    """Configuration for market data sources."""
    # Market data sources
    primary_source: str = ""
    secondary_sources: List[str] = field(default_factory=list)

    # Data quality settings
    require_consolidated_feed: bool = False
    validate_data_integrity: bool = True

    # Subscription settings
    subscribe_to_level1: bool = True
    subscribe_to_level2: bool = False
    subscribe_to_trades: bool = True

    # Throttling settings
    max_update_frequency_ms: int = 100

    def validate(self) -> List[str]:
        """Validate the market data configuration."""
        errors = super().validate()

        # Validate primary source
        if not self.primary_source:
            errors.append("primary_source must not be empty")

        # Validate update frequency
        if self.max_update_frequency_ms <= 0:
            errors.append("max_update_frequency_ms must be positive")

        return errors


@dataclass
class ExchangeConfig(BaseConfig):
    """
    Main exchange configuration that contains all exchange-related settings.

    This class serves as a container for exchange instances, connection settings,
    and other exchange-related configurations.
    """
    # Exchange instances
    instances: Dict[str, ExchangeInstanceConfig] = field(default_factory=dict)

    # Default exchange settings
    default_exchange_id: str = ""

    # Webhook configuration
    webhooks: WebhookConfig = field(default_factory=WebhookConfig)

    # Order routing configuration
    order_routing: OrderRoutingConfig = field(default_factory=OrderRoutingConfig)

    # Market data configuration
    market_data: MarketDataConfig = field(default_factory=MarketDataConfig)

    # Connection management
    connection_check_interval_sec: int = 60
    max_connection_failures: int = 3
    reconnect_delay_ms: int = 5000

    # Security settings
    store_credentials_securely: bool = True
    rotate_keys_interval_days: int = 90

    def __post_init__(self):
        """Initialize the exchange configuration."""
        super().__post_init__()

        # Process instances dictionary
        self._instance_configs = {}
        for instance_id, instance_dict in self.instances.items():
            if isinstance(instance_dict, dict):
                instance_config = ExchangeInstanceConfig(**instance_dict)
                instance_config.instance_id = instance_id
                self._instance_configs[instance_id] = instance_config
            elif isinstance(instance_dict, ExchangeInstanceConfig):
                self._instance_configs[instance_id] = instance_dict

        # Ensure webhooks is a WebhookConfig object
        if isinstance(self.webhooks, dict):
            self.webhooks = WebhookConfig(**self.webhooks)

        # Ensure order_routing is an OrderRoutingConfig object
        if isinstance(self.order_routing, dict):
            self.order_routing = OrderRoutingConfig(**self.order_routing)

        # Ensure market_data is a MarketDataConfig object
        if isinstance(self.market_data, dict):
            self.market_data = MarketDataConfig(**self.market_data)

        # Set default_exchange_id to the first instance if not specified
        if not self.default_exchange_id and self._instance_configs:
            self.default_exchange_id = next(iter(self._instance_configs.keys()))

        # Set primary_exchange in order_routing to default_exchange_id if not specified
        if self.order_routing.enable_smart_routing and not self.order_routing.primary_exchange:
            self.order_routing.primary_exchange = self.default_exchange_id

        # Set primary_source in market_data to default_exchange_id if not specified
        if not self.market_data.primary_source:
            self.market_data.primary_source = self.default_exchange_id

    def get_instance(self, instance_id: str) -> Optional[ExchangeInstanceConfig]:
        """
        Get an exchange instance configuration by ID.

        Args:
            instance_id: The ID of the exchange instance

        Returns:
            The exchange instance configuration if found, None otherwise
        """
        return self._instance_configs.get(instance_id)

    def get_default_instance(self) -> Optional[ExchangeInstanceConfig]:
        """
        Get the default exchange instance configuration.

        Returns:
            The default exchange instance configuration if set, None otherwise
        """
        if self.default_exchange_id:
            return self._instance_configs.get(self.default_exchange_id)
        return None

    def get_instances_by_provider(self, provider: ExchangeProvider) -> List[ExchangeInstanceConfig]:
        """
        Get exchange instance configurations by provider.

        Args:
            provider: The exchange provider to filter by

        Returns:
            A list of exchange instance configurations for the specified provider
        """
        return [
            instance for instance in self._instance_configs.values()
            if instance.provider == provider
        ]

    def get_instances_by_asset_class(self, asset_class: AssetClass) -> List[ExchangeInstanceConfig]:
        """
        Get exchange instance configurations by asset class.

        Args:
            asset_class: The asset class to filter by

        Returns:
            A list of exchange instance configurations that support the specified asset class
        """
        return [
            instance for instance in self._instance_configs.values()
            if asset_class in instance.asset_classes
        ]

    def add_instance(self, instance_config: ExchangeInstanceConfig) -> None:
        """
        Add an exchange instance configuration.

        Args:
            instance_config: The exchange instance configuration to add
        """
        instance_id = instance_config.instance_id
        if not instance_id:
            raise ValueError("Instance ID cannot be empty")

        self._instance_configs[instance_id] = instance_config
        self.instances[instance_id] = instance_config.to_dict()

    def remove_instance(self, instance_id: str) -> bool:
        """
        Remove an exchange instance configuration.

        Args:
            instance_id: The ID of the exchange instance to remove

        Returns:
            True if the instance was removed, False otherwise
        """
        if instance_id in self._instance_configs:
            del self._instance_configs[instance_id]
            del self.instances[instance_id]

            # If we removed the default exchange, update it
            if instance_id == self.default_exchange_id:
                if self._instance_configs:
                    self.default_exchange_id = next(iter(self._instance_configs.keys()))
                else:
                    self.default_exchange_id = ""

            return True
        return False

    def validate(self) -> List[str]:
        """Validate the exchange configuration."""
        errors = super().validate()

        # Validate instances
        if not self._instance_configs:
            errors.append("At least one exchange instance must be configured")

        for instance_id, config in self._instance_configs.items():
            instance_errors = config.validate()
            for error in instance_errors:
                errors.append(f"In instance '{instance_id}': {error}")

        # Validate default_exchange_id
        if self.default_exchange_id and self.default_exchange_id not in self._instance_configs:
            errors.append(f"default_exchange_id '{self.default_exchange_id}' does not exist")

        # Validate webhooks
        webhook_errors = self.webhooks.validate()
        for error in webhook_errors:
            errors.append(f"In webhooks: {error}")

        # Validate order_routing
        routing_errors = self.order_routing.validate()
        for error in routing_errors:
            errors.append(f"In order_routing: {error}")

        # Validate market_data
        market_data_errors = self.market_data.validate()
        for error in market_data_errors:
            errors.append(f"In market_data: {error}")

        # Validate order_routing references
        if self.order_routing.enable_smart_routing:
            if self.order_routing.primary_exchange not in self._instance_configs:
                errors.append(f"primary_exchange '{self.order_routing.primary_exchange}' in order_routing does not exist")

            for backup_exchange in self.order_routing.backup_exchanges:
                if backup_exchange not in self._instance_configs:
                    errors.append(f"backup_exchange '{backup_exchange}' in order_routing does not exist")

        # Validate market_data references
        if self.market_data.primary_source not in self._instance_configs:
            errors.append(f"primary_source '{self.market_data.primary_source}' in market_data does not exist")

        for secondary_source in self.market_data.secondary_sources:
            if secondary_source not in self._instance_configs:
                errors.append(f"secondary_source '{secondary_source}' in market_data does not exist")

        # Validate connection settings
        if self.connection_check_interval_sec <= 0:
            errors.append("connection_check_interval_sec must be positive")

        if self.max_connection_failures <= 0:
            errors.append("max_connection_failures must be positive")

        if self.reconnect_delay_ms <= 0:
            errors.append("reconnect_delay_ms must be positive")

        # Validate security settings
        if self.rotate_keys_interval_days <= 0:
            errors.append("rotate_keys_interval_days must be positive")

        return errors


def get_exchange_config(config_path: Optional[Union[str, Path]] = None) -> ExchangeConfig:
    """
    Get the exchange configuration.

    Args:
        config_path: Optional path to a configuration file. If not provided,
                    the default path from the ConfigManager will be used.

    Returns:
        The exchange configuration.
    """
    if config_path is None:
        config_path = ConfigManager.get_config_path("exchange")

    return ConfigManager.load_config(
        ExchangeConfig,
        config_path=config_path,
        env_prefix="TRADING_EXCHANGE",
        reload=False
    )