#!/usr/bin/env python3
"""
Periodic Boundary Condition (PBC) utilities for 1/8 motor models.

Provides:
    identify_boundary_nodes()  -- find master/slave nodes by angular position
    find_pbc_edge_pairs()      -- KDTree-based matching → bidirectional PBC edges

Anti-periodic symmetry for 2D magnetostatics (1/8 model):
    A_slave = -A_master   (magnetic vector potential reverses sign)
    Edge attr flag: normal interior edges = 1.0, anti-periodic PBC edges = -1.0
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from scipy.spatial import KDTree


# ---------------------------------------------------------------------------
# Boundary node identification
# ---------------------------------------------------------------------------

def identify_boundary_nodes(
    pos_x: np.ndarray,
    pos_y: np.ndarray,
    master_angle_deg: float = 0.0,
    slave_angle_deg: float = 45.0,
    angle_tol_deg: float = 0.5,
    r_min: float = 1e-4,
) -> Tuple[np.ndarray, np.ndarray]:
    """Identify master and slave boundary nodes by angular position.

    Args:
        pos_x, pos_y:      Node coordinates (mm or any consistent unit).
        master_angle_deg:  Expected angle of master boundary (degrees).
        slave_angle_deg:   Expected angle of slave boundary (degrees).
        angle_tol_deg:     Angular tolerance for node selection (degrees).
        r_min:             Minimum radius — ignores nodes at/near the origin.

    Returns:
        master_indices:  int array of node indices on the master boundary.
        slave_indices:   int array of node indices on the slave boundary.
    """
    r = np.hypot(pos_x, pos_y)
    valid = r > r_min

    angles_deg = np.degrees(np.arctan2(pos_y, pos_x))

    # Angular difference modulo 360 → clamp to [-180, 180]
    def _angle_diff(a, ref):
        d = (a - ref + 180.0) % 360.0 - 180.0
        return np.abs(d)

    master_mask = valid & (_angle_diff(angles_deg, master_angle_deg) < angle_tol_deg)
    slave_mask = valid & (_angle_diff(angles_deg, slave_angle_deg) < angle_tol_deg)

    return np.where(master_mask)[0], np.where(slave_mask)[0]


# ---------------------------------------------------------------------------
# PBC edge pair matching
# ---------------------------------------------------------------------------

def find_pbc_edge_pairs(
    pos_x: np.ndarray,
    pos_y: np.ndarray,
    master_indices: np.ndarray,
    slave_indices: np.ndarray,
    rotation_deg: float = -45.0,
    tol: float = 1e-3,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Find matching node pairs between master and slave boundaries using KDTree.

    Strategy:
        1. Rotate slave node coordinates by `rotation_deg` to align with master.
        2. For each rotated-slave node, find the nearest master node within `tol`.
        3. Create bidirectional PBC edge_index (master→slave and slave→master).
        4. Assign edge attr flag = -1.0 for all PBC edges (anti-periodic).

    Args:
        pos_x, pos_y:     All node coordinates (full mesh).
        master_indices:   Node indices of master boundary nodes.
        slave_indices:    Node indices of slave boundary nodes.
        rotation_deg:     Rotation applied to slave nodes to align with master.
                          For 1/8 model (45° sector): -45.0.
        tol:              KDTree lookup tolerance (same unit as coordinates).

    Returns:
        pbc_edge_index:  LongTensor [2, 2*N_matched]  bidirectional edge index.
        pbc_edge_attr:   FloatTensor [2*N_matched, 1]  all entries = -1.0.
    """
    if len(master_indices) == 0 or len(slave_indices) == 0:
        empty_idx = torch.zeros((2, 0), dtype=torch.long)
        empty_attr = torch.zeros((0, 1), dtype=torch.float32)
        return empty_idx, empty_attr

    # Coordinates of master and slave nodes
    master_xy = np.column_stack([pos_x[master_indices], pos_y[master_indices]])  # (M, 2)
    slave_xy = np.column_stack([pos_x[slave_indices], pos_y[slave_indices]])     # (S, 2)

    # Rotate slave nodes so they align spatially with master nodes
    theta = np.radians(rotation_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    rot = np.array([[cos_t, -sin_t], [sin_t, cos_t]])  # (2, 2)
    slave_xy_rot = slave_xy @ rot.T  # (S, 2) — each row is rotated

    # KDTree search: for each rotated-slave node, find nearest master node
    tree = KDTree(master_xy)
    dists, master_lut = tree.query(slave_xy_rot, k=1)  # (S,), (S,)

    # Keep only matches within tolerance
    valid = dists < tol
    if not np.any(valid):
        empty_idx = torch.zeros((2, 0), dtype=torch.long)
        empty_attr = torch.zeros((0, 1), dtype=torch.float32)
        return empty_idx, empty_attr

    # Map back to global node indices
    global_master = master_indices[master_lut[valid]]  # (K,)
    global_slave = slave_indices[np.where(valid)[0]]   # (K,)

    # Bidirectional edges: master→slave and slave→master
    src = np.concatenate([global_master, global_slave])  # (2K,)
    dst = np.concatenate([global_slave, global_master])  # (2K,)

    pbc_edge_index = torch.tensor(
        np.stack([src, dst], axis=0), dtype=torch.long
    )  # [2, 2K]
    n_edges = pbc_edge_index.shape[1]
    pbc_edge_attr = torch.full((n_edges, 1), -1.0, dtype=torch.float32)  # anti-periodic

    return pbc_edge_index, pbc_edge_attr
