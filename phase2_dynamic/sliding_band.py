"""Pure sliding-band edge builder for dynamic rotor-gap topology.

Contract (from phase2_dynamic_edge_pipeline.md):
  - Pure function: no side effects, no global state writes
  - Compatible with static_edge_index concatenation
  - Must not add duplicate edges with static topology
"""

from __future__ import annotations

import numpy as np

try:
    from scipy.spatial import KDTree
except ImportError as e:
    raise ImportError("scipy required for build_sliding_band_edges") from e


def build_sliding_band_edges(
    pos: np.ndarray,
    gap_mask: np.ndarray,
    rotor_angle_deg: float,
    stator_indices: np.ndarray,
    rotor_indices: np.ndarray,
    *,
    k_neighbors: int = 3,
    max_dist_mm: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Build dynamic sliding-band edges for a given rotor angle.

    Only connects nodes within the gap region (gap_mask == True) to their
    k nearest neighbors across the stator/rotor boundary.

    Args:
        pos:              Node positions [N, 2] float32, mm
        gap_mask:         Boolean mask [N], True for gap-region nodes
        rotor_angle_deg:  Current rotor angle in degrees
        stator_indices:   Indices of stator-side gap nodes
        rotor_indices:    Indices of rotor-side gap nodes
        k_neighbors:      Number of nearest neighbors to connect
        max_dist_mm:      Maximum connection distance (mm)

    Returns:
        edge_index:  [2, E_dyn] int64, bidirectional
        edge_attr:   [E_dyn, 1] float32, interior sign = +1.0
    """
    if len(stator_indices) == 0 or len(rotor_indices) == 0:
        return (
            np.empty((2, 0), dtype=np.int64),
            np.empty((0, 1), dtype=np.float32),
        )

    # Rotate rotor nodes by current angle
    theta = np.deg2rad(rotor_angle_deg)
    rot = np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta),  np.cos(theta)]], dtype=np.float32)
    rotor_pos_rotated = (rot @ pos[rotor_indices].T).T   # [R, 2]
    stator_pos = pos[stator_indices]                      # [S, 2]

    # KDTree: stator as target, query from rotor
    tree = KDTree(stator_pos)
    dists, nn_idx = tree.query(rotor_pos_rotated, k=k_neighbors, workers=1)
    # dists: [R, k], nn_idx: [R, k]

    src_list, dst_list = [], []
    for r_local, (dist_row, nn_row) in enumerate(zip(dists, nn_idx)):
        r_global = int(rotor_indices[r_local])
        for k_i, (d, s_local) in enumerate(zip(dist_row, nn_row)):
            if d > max_dist_mm:
                continue
            s_global = int(stator_indices[s_local])
            src_list.append(r_global)
            dst_list.append(s_global)
            src_list.append(s_global)   # bidirectional
            dst_list.append(r_global)

    if not src_list:
        return (
            np.empty((2, 0), dtype=np.int64),
            np.empty((0, 1), dtype=np.float32),
        )

    edge_index = np.array([src_list, dst_list], dtype=np.int64)
    # Deduplicate
    edge_set = np.unique(edge_index, axis=1)
    edge_attr = np.ones((edge_set.shape[1], 1), dtype=np.float32)

    return edge_set, edge_attr
