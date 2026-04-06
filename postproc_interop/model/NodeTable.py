from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NodeTable:
    """Immutable node ids and 2D coordinates in millimeters."""

    node_id: np.ndarray
    x_mm: np.ndarray
    y_mm: np.ndarray

    def __post_init__(self) -> None:
        lengths = {
            int(self.node_id.shape[0]),
            int(self.x_mm.shape[0]),
            int(self.y_mm.shape[0]),
        }
        if len(lengths) != 1:
            raise ValueError("NodeTable arrays must share the same length")

        for array in (self.node_id, self.x_mm, self.y_mm):
            array.flags.writeable = False
