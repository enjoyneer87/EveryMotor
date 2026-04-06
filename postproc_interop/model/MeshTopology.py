from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MeshTopology:
    """Immutable triangular mesh connectivity and region assignment."""

    tri_index: np.ndarray
    node_1: np.ndarray
    node_2: np.ndarray
    node_3: np.ndarray
    reg_code: np.ndarray

    def __post_init__(self) -> None:
        lengths = {
            int(self.tri_index.shape[0]),
            int(self.node_1.shape[0]),
            int(self.node_2.shape[0]),
            int(self.node_3.shape[0]),
            int(self.reg_code.shape[0]),
        }
        if len(lengths) != 1:
            raise ValueError("MeshTopology arrays must share the same length")

        for array in (
            self.tri_index,
            self.node_1,
            self.node_2,
            self.node_3,
            self.reg_code,
        ):
            array.flags.writeable = False
