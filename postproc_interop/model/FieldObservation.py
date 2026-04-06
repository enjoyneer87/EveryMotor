from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


FieldAssociation = Literal["node", "element"]


@dataclass(frozen=True)
class FieldObservation:
    """Immutable field observation with explicit association semantics."""

    name: str
    values: np.ndarray
    association: FieldAssociation
    unit: str = ""
    source_key: str = ""

    def __post_init__(self) -> None:
        if self.association not in {"node", "element"}:
            raise ValueError(
                "association must be 'node' or 'element', got "
                f"{self.association}"
            )
        self.values.flags.writeable = False

    @property
    def sample_count(self) -> int:
        if self.values.ndim == 0:
            return 1
        return int(self.values.shape[-1])
