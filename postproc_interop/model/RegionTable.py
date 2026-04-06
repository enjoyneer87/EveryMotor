from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from postproc_interop.model.FrozenMapping import FrozenMapping


@dataclass(frozen=True)
class RegionTable:
    """Immutable region code-name lookup and optional metadata."""

    reg_code: np.ndarray
    name: np.ndarray
    meta: FrozenMapping = field(default_factory=FrozenMapping)

    def __post_init__(self) -> None:
        if int(self.reg_code.shape[0]) != int(self.name.shape[0]):
            raise ValueError("RegionTable arrays must share the same length")
        self.reg_code.flags.writeable = False
        self.name.flags.writeable = False
        if not isinstance(self.meta, FrozenMapping):
            object.__setattr__(self, "meta", FrozenMapping(self.meta))

    @classmethod
    def from_mapping(
        cls,
        *,
        reg_code: np.ndarray,
        name: np.ndarray,
        meta: Mapping[str, Any] | None = None,
    ) -> "RegionTable":
        return cls(
            reg_code=reg_code,
            name=name,
            meta=FrozenMapping(meta),
        )
