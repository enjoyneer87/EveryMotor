"""Data preprocessing utilities for the Phase 1 static 1/8 motor model.

- Build periodic/anti-periodic boundary edges with KDTree matching.
- Provide helpers to deduplicate interior edges from element connectivity.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import numpy as np
import torch

try:
    from scipy.spatial import KDTree
except ImportError as exc:  # pragma: no cover - scipy not always installed in CI
    raise ImportError(
        "scipy is required for KDTree-based periodic boundary matching. "
        "Install scipy before running Phase 1 preprocessing."
    ) from exc


LOG = logging.getLogger(__name__)


@dataclass
class PBCMatchResult:
    """Container for periodic boundary matching diagnostics."""

    matched_master: np.ndarray
    matched_slave: np.ndarray
    unmatched_slave: np.ndarray
    distances: np.ndarray


def rotate_points(
    xy: np.ndarray,
    angle_deg: float,
    origin_xy: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Rotate 2D points by ``angle_deg`` degrees around ``origin_xy``."""
    rad = math.radians(angle_deg)
    c, s = math.cos(rad), math.sin(rad)
    rot = np.array([[c, -s], [s, c]], dtype=np.float64)
    pts = np.asarray(xy, dtype=np.float64)
    if origin_xy is None:
        origin = np.zeros((2,), dtype=np.float64)
    else:
        origin = np.asarray(origin_xy, dtype=np.float64)
    return (pts - origin) @ rot.T + origin


def match_periodic_boundary(
    master_nodes: np.ndarray,
    slave_nodes: np.ndarray,
    angle_deg: float = -45.0,
    atol: float = 1e-5,
    rtol: float = 1e-8,
    origin_xy: Optional[np.ndarray] = None,
) -> PBCMatchResult:
    """Match master/slave boundary nodes for the 1/8 model.

    Rotates the slave nodes, then finds nearest master nodes within tolerance.
    Returns indices (0-based) into the provided arrays.
    """
    master = np.asarray(master_nodes, dtype=np.float64)
    slave = rotate_points(
        np.asarray(slave_nodes, dtype=np.float64),
        angle_deg,
        origin_xy=origin_xy,
    )

    tree = KDTree(master)
    dist, idx = tree.query(slave, distance_upper_bound=atol)

    finite = np.isfinite(dist)
    valid_idx = finite & (idx < master.shape[0])

    rel_scale = np.zeros(slave.shape[0], dtype=np.float64)
    if np.any(valid_idx):
        rel_scale[valid_idx] = np.max(
            np.abs(master[idx[valid_idx]] - slave[valid_idx]),
            axis=1,
        )

    within_tol = valid_idx & (dist <= atol + rtol * rel_scale)
    matched_slave = np.nonzero(within_tol)[0]
    matched_master = idx[within_tol].astype(np.int64, copy=False)
    unmatched_slave = np.nonzero(~within_tol)[0]

    if unmatched_slave.size > 0:
        LOG.warning(
            "PBC matching skipped %d slave nodes outside tolerance (atol=%g, rtol=%g)",
            unmatched_slave.size,
            atol,
            rtol,
        )

    return PBCMatchResult(
        matched_master=matched_master,
        matched_slave=matched_slave,
        unmatched_slave=unmatched_slave,
        distances=dist,
    )


def dedupe_edges_from_elements(element_triples: np.ndarray) -> np.ndarray:
    """Build undirected, deduplicated interior edges from triangular elements."""
    elems = np.asarray(element_triples, dtype=np.int64)
    if elems.ndim != 2 or elems.shape[1] != 3:
        raise ValueError("element_triples must have shape (n_elem, 3)")

    i1, i2, i3 = elems[:, 0], elems[:, 1], elems[:, 2]
    edges = np.stack(
        [
            np.stack([i1, i2], axis=1),
            np.stack([i2, i3], axis=1),
            np.stack([i3, i1], axis=1),
        ],
        axis=0,
    ).reshape(-1, 2)

    # Sort each edge to treat as undirected, then drop duplicates.
    edges = np.sort(edges, axis=1)
    uniq = np.unique(edges, axis=0)
    return uniq.astype(np.int64, copy=False)


def build_pbc_edges(
    master_nodes: np.ndarray,
    slave_nodes: np.ndarray,
    angle_deg: float = -45.0,
    atol: float = 1e-5,
    rtol: float = 1e-8,
    anti_periodic: bool = True,
    origin_xy: Optional[np.ndarray] = None,
) -> Tuple[torch.Tensor, torch.Tensor, PBCMatchResult]:
    """Create bidirectional PBC edge_index and edge_attr tensors."""
    match = match_periodic_boundary(
        master_nodes,
        slave_nodes,
        angle_deg=angle_deg,
        atol=atol,
        rtol=rtol,
        origin_xy=origin_xy,
    )
    master_idx = match.matched_master
    slave_idx = match.matched_slave
    n_edges = master_idx.size

    if n_edges == 0:
        raise ValueError("No PBC edges were created; check tolerance or node sets.")

    edges = np.column_stack([master_idx, slave_idx]).astype(np.int64, copy=False)
    edges_bidir = np.concatenate([edges, edges[:, ::-1]], axis=0)
    edge_index = torch.from_numpy(edges_bidir.T.copy()).contiguous()

    sign = -1.0 if anti_periodic else 1.0
    edge_attr = torch.full((edge_index.shape[1], 1), sign, dtype=torch.float32)
    return edge_index, edge_attr, match


def combine_edges(
    interior_edge_index: torch.Tensor,
    pbc_edge_index: torch.Tensor,
    pbc_edge_attr: torch.Tensor,
    interior_attr_value: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Concatenate interior and PBC edges, ensuring edge_attr has shape [E, 1]."""
    if interior_edge_index.dim() != 2 or interior_edge_index.size(0) != 2:
        raise ValueError("interior_edge_index must have shape [2, num_edges]")
    if pbc_edge_index.dim() != 2 or pbc_edge_index.size(0) != 2:
        raise ValueError("pbc_edge_index must have shape [2, num_edges]")

    interior_attr = torch.full(
        (interior_edge_index.shape[1], 1),
        float(interior_attr_value),
        dtype=torch.float32,
        device=interior_edge_index.device,
    )

    edge_index = torch.cat([interior_edge_index, pbc_edge_index], dim=1).contiguous()
    edge_attr = torch.cat([interior_attr, pbc_edge_attr], dim=0).contiguous()
    return edge_index, edge_attr


def to_tensor(x: Iterable[float], dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Convert input to a torch tensor with consistent dtype."""
    return torch.as_tensor(x, dtype=dtype)

