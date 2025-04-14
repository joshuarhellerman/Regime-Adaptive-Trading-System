"""
fill_model.py - Immutable representation of order fills

This module defines the core Fill class for representing trade executions
in the system. Fills are immutable records of order execution events.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional, Any, Union
from decimal import Decimal
from types import MappingProxyType

class Fill:
    """
    Immutable representation of an order fill.

    A Fill represents a trade execution, either partial or complete,
    for an order. Fills are immutable to ensure audit trail integrity.
    """

    def __init__(self,
                order_id: str,
                fill_id: Optional[str] = None,
                timestamp: Optional[datetime] = None,
                instrument: str = "",
                quantity: Union[float, Decimal] = 0.0,
                price: Union[float, Decimal] = 0.0,
                fees: Union[float, Decimal] = 0.0,
                exchange_id: Optional[str] = None,
                is_maker: bool = False,
                exchange_fill_id: Optional[str] = None,
                metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize a fill object.

        Args:
            order_id: ID of the order that was filled
            fill_id: Unique ID for this fill (generated if not provided)
            timestamp: Time of the fill (current time if not provided)
            instrument: Trading instrument symbol
            quantity: Amount that was filled
            price: Price at which the fill occurred
            fees: Trading fees for this fill
            exchange_id: ID of the exchange where the fill occurred
            is_maker: Whether this fill was a maker (vs taker)
            exchange_fill_id: Fill ID assigned by the exchange
            metadata: Additional data about the fill
        """
        # Make all fields immutable
        self._data = {
            "order_id": order_id,
            "fill_id": fill_id or str(uuid.uuid4()),
            "timestamp": timestamp or datetime.utcnow(),
            "instrument": instrument,
            "quantity": Decimal(str(quantity)),
            "price": Decimal(str(price)),
            "fees": Decimal(str(fees)),
            "exchange_id": exchange_id,
            "is_maker": is_maker,
            "exchange_fill_id": exchange_fill_id,
            "metadata": metadata or {}
        }

        # Freeze the metadata dictionary
        self._data["metadata"] = MappingProxyType(self._data["metadata"])

    def __getattr__(self, name):
        """Allow attribute access to internal data dictionary."""
        if name in self._data:
            return self._data[name]
        raise AttributeError(f"'Fill' object has no attribute '{name}'")

    def with_updates(self, **kwargs) -> 'Fill':
        """
        Create a new Fill with updated values.

        Args:
            **kwargs: Values to update in the new Fill

        Returns:
            A new Fill instance with updated values
        """
        new_data = dict(self._data)
        new_data.update(kwargs)

        # Create new instance
        return Fill(
            order_id=new_data["order_id"],
            fill_id=new_data["fill_id"],
            timestamp=new_data["timestamp"],
            instrument=new_data["instrument"],
            quantity=new_data["quantity"],
            price=new_data["price"],
            fees=new_data["fees"],
            exchange_id=new_data["exchange_id"],
            is_maker=new_data["is_maker"],
            exchange_fill_id=new_data["exchange_fill_id"],
            metadata=dict(new_data["metadata"])
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert fill to dictionary for serialization.

        Returns:
            Dict containing all fill data
        """
        result = dict(self._data)

        # Convert non-serializable types
        result["timestamp"] = result["timestamp"].isoformat()
        result["quantity"] = float(result["quantity"])
        result["price"] = float(result["price"])
        result["fees"] = float(result["fees"])
        result["metadata"] = dict(result["metadata"])

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Fill':
        """
        Create a fill from a dictionary.

        Args:
            data: Dictionary with fill data

        Returns:
            Fill instance
        """
        # Convert timestamp string to datetime
        if isinstance(data.get("timestamp"), str):
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])

        return cls(**data)

    def __eq__(self, other):
        """Compare fills based on their data."""
        if not isinstance(other, Fill):
            return False
        return self._data == other._data

    def __repr__(self):
        """String representation of the fill."""
        return (f"Fill(order_id={self.order_id}, "
                f"quantity={float(self.quantity)}, "
                f"price={float(self.price)}, "
                f"timestamp={self.timestamp.isoformat()})")