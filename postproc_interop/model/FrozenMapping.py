from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any


class FrozenMapping(Mapping[str, Any]):
    """Immutable mapping wrapper for canonical contract payloads."""

    def __init__(self, data: Mapping[str, Any] | None = None) -> None:
        self._data = dict(data or {})

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return f"FrozenMapping({self._data!r})"

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def to_dict(self) -> dict[str, Any]:
        return dict(self._data)
