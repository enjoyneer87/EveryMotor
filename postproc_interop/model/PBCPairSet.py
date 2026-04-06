from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PBCPairSet:
    """Immutable matched PBC node pairs and diagnostics."""

    pbc_forward_index: np.ndarray
    match_ratio: float
    match_mode: str
    max_rotation_residual: float
    anti_periodic: bool
    error_code: str = ""

    def __post_init__(self) -> None:
        if (
            self.pbc_forward_index.ndim != 2
            or self.pbc_forward_index.shape[0] != 2
        ):
            raise ValueError(
                "pbc_forward_index must have shape [2, P], "
                f"got {self.pbc_forward_index.shape}"
            )
        if not (0.0 <= float(self.match_ratio) <= 1.0):
            raise ValueError("match_ratio must be in [0.0, 1.0]")
        self.pbc_forward_index.flags.writeable = False
