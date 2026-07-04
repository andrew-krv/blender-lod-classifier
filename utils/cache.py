"""Small in-memory cache helper."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SimpleCache:
    """Dictionary-backed cache for one addon session."""

    _storage: dict[str, Any] = field(default_factory=dict)

    def get(self, key: str) -> Any:
        """Return value for key or None when missing."""

        return self._storage.get(key)

    def set(self, key: str, value: Any) -> None:
        """Store value by key."""

        self._storage[key] = value

    def delete(self, key: str) -> None:
        """Delete key if present."""

        self._storage.pop(key, None)

    def clear(self) -> None:
        """Remove all cached entries."""

        self._storage.clear()
