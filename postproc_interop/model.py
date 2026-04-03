from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class MeshTopology:
    """Triangular element connectivity and region assignment."""

    tri_index: np.ndarray
    node_1: np.ndarray
    node_2: np.ndarray
    node_3: np.ndarray
    reg_code: np.ndarray


@dataclass
class NodeTable:
    """Node indices and coordinates."""

    node_id: np.ndarray
    x_mm: np.ndarray
    y_mm: np.ndarray


@dataclass
class RegionTable:
    """Region code-name mapping and optional metadata."""

    reg_code: np.ndarray
    name: np.ndarray
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class MeshFrame:
    """Canonical representation independent of source file format.

    Fields are optional because some sources contain only subset data.
    """

    topology: MeshTopology
    nodes: NodeTable
    regions: RegionTable | None = None
    fields: dict[str, np.ndarray] = field(default_factory=dict)
    attrs: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        return {
            "n_elements": int(self.topology.tri_index.shape[0]),
            "n_nodes": int(self.nodes.node_id.shape[0]),
            "field_keys": sorted(self.fields.keys()),
            "has_regions": self.regions is not None,
        }
