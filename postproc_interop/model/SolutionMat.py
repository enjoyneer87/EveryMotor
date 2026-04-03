from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from postproc_interop.model.Solution import Solution


@dataclass
class SolutionMat(Solution):
    """Numpy array-based concrete solution (cf. pyleecan SolutionMat).

    Stores one physical field (e.g. Bx, By, A, J) as a 1-D numpy array
    aligned to either elements or nodes.
    """

    label: str          # e.g. "Bx", "By", "A", "J"
    field: np.ndarray   # shape (E,) per-element  or  (N,) per-node  [float]
    unit: str = ""      # e.g. "T", "Wb/m", "A/mm2"
    type_element: str = "triangle"  # "triangle" | "node"
