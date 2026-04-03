from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from postproc_interop.model.Mesh import Mesh


@dataclass
class MeshMat(Mesh):
    """Numpy array-based concrete mesh (cf. pyleecan MeshMat).

    Stores triangular FEA mesh: nodes, element connectivity, and region mapping.
    Field data (Bx, By, A, J) is NOT stored here — use MeshSolution.solution_dict.
    """

    # ── nodes ────────────────────────────────────────────────────
    node_id: np.ndarray   # shape (N,)  int
    x_mm: np.ndarray      # shape (N,)  float  [mm]
    y_mm: np.ndarray      # shape (N,)  float  [mm]

    # ── triangular element connectivity ─────────────────────────
    tri_index: np.ndarray  # shape (E,)  int
    node_1: np.ndarray     # shape (E,)  int
    node_2: np.ndarray     # shape (E,)  int
    node_3: np.ndarray     # shape (E,)  int
    reg_code: np.ndarray   # shape (E,)  int  — per-element region code

    # ── region table (optional) ──────────────────────────────────
    region_code: np.ndarray | None = None  # shape (R,)  int
    region_name: np.ndarray | None = None  # shape (R,)  dtype=object (str or bytes)

    # ── metadata ─────────────────────────────────────────────────
    attrs: dict[str, Any] = field(default_factory=dict)
