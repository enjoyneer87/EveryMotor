"""Phase 1 periodic boundary pairing from chain families.

Morphism: (BoundaryChainSet, PriorContext, nodes_xy) → PeriodicPairSet (immutable)

Uses scipy.spatial.KDTree for node matching (replicates match_periodic_boundary
from data_preprocessing.py without torch dependency).
"""

from __future__ import annotations

import logging
import math
from typing import List, Tuple

import numpy as np
from scipy.spatial import KDTree

from phase1_static.pbc_contracts import (
    BoundaryChainSet,
    PriorContext,
    PeriodicChainPair,
    PeriodicPairSet,
    PeriodicPairSetResult,
)


LOG = logging.getLogger(__name__)

# Matching tolerance for spatial alignment
SPATIAL_MATCH_TOL = 1e-3  # meters


def _rotate_points(xy: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate 2D points by angle_deg (counter-clockwise)."""
    rad = math.radians(angle_deg)
    c, s = math.cos(rad), math.sin(rad)
    rot = np.array([[c, -s], [s, c]], dtype=np.float64)
    return np.asarray(xy, dtype=np.float64) @ rot.T


def _match_periodic_nodes(
    master_nodes: np.ndarray,
    slave_nodes: np.ndarray,
    angle_deg: float = -45.0,
    atol: float = 1e-5,
) -> Tuple[List[int], List[int]]:
    """Match slave nodes to master nodes after rotation.

    Returns:
        (matched_master_indices, matched_slave_indices)
    """
    master = np.asarray(master_nodes, dtype=np.float64)
    slave = _rotate_points(np.asarray(slave_nodes, dtype=np.float64), angle_deg)

    if len(master) == 0 or len(slave) == 0:
        return [], []

    tree = KDTree(master)
    dists, indices = tree.query(slave)

    matched_master_idx = []
    matched_slave_idx = []

    for i, (dist, idx) in enumerate(zip(dists, indices)):
        if dist <= atol:
            matched_master_idx.append(int(idx))
            matched_slave_idx.append(int(i))

    return matched_master_idx, matched_slave_idx


def build_periodic_pair_set(
    chain_set: BoundaryChainSet,
    prior: PriorContext,
    nodes_xy: np.ndarray,
) -> PeriodicPairSetResult:
    """Build periodic chain pair set from boundary chains and prior.

    Pure morphism: (BoundaryChainSet, PriorContext, nodes_xy) → PeriodicPairSetResult

    Strategy:
      1. Expected rotation = 360 / n_geom_prior
      2. For each chain family (by type), propose master-slave pairs
      3. Use match_periodic_boundary for KDTree spatial validation
      4. Materialize pbc_edge_index [2, 2P] and pbc_edge_attr [2P, 1]

    Args:
        chain_set: Extracted boundary chains
        prior: Inferred global prior (n_geom_prior, anti_periodic_prior)
        nodes_xy: Node coordinates for chain endpoints

    Returns:
        (pair_set, error_code) tuple
    """
    if not chain_set.chains:
        return None, "E-PBC-PAIR-001"

    expected_rotation_deg = 360.0 / prior.n_geom_prior

    # Group chains by type
    chains_by_type = {}
    for idx, chain in enumerate(chain_set.chains):
        if chain.chain_type not in chains_by_type:
            chains_by_type[chain.chain_type] = []
        chains_by_type[chain.chain_type].append((idx, chain))

    # Propose pairs within each family
    candidate_pairs = []

    for chain_type, chain_list in chains_by_type.items():
        if len(chain_list) < 2:
            # Single chain of this type, cannot pair
            continue

        # For simplicity: pair chain 0 with chain 1 as master-slave
        # In real scenarios, would use spatial rotation to find nearest neighbor
        idx_0, chain_0 = chain_list[0]
        idx_1, chain_1 = chain_list[1]

        # Get chain endpoint nodes
        master_idx_0, master_idx_1 = chain_0.endpoint_pair
        slave_idx_0, slave_idx_1 = chain_1.endpoint_pair

        master_endpoints = nodes_xy[[int(master_idx_0), int(master_idx_1)]]
        slave_endpoints = nodes_xy[[int(slave_idx_0), int(slave_idx_1)]]

        # Try to match via rotation
        matched_master, matched_slave = _match_periodic_nodes(
            master_nodes=master_endpoints,
            slave_nodes=slave_endpoints,
            angle_deg=-expected_rotation_deg,
            atol=SPATIAL_MATCH_TOL,
        )

        # Check if match is meaningful
        if len(matched_slave) > 0:
            match_ratio = len(matched_slave) / len(slave_endpoints)
            pair = PeriodicChainPair(
                master_chain_idx=idx_0,
                slave_chain_idx=idx_1,
                rotation_deg=-expected_rotation_deg,
                match_ratio=match_ratio,
            )
            candidate_pairs.append(pair)

    if not candidate_pairs:
        return None, "E-PBC-PAIR-001"

    # Materialize edge_index and edge_attr
    all_master_idx = []
    all_slave_idx = []
    all_attrs = []

    for pair in candidate_pairs:
        master_chain = chain_set.chains[pair.master_chain_idx]
        slave_chain = chain_set.chains[pair.slave_chain_idx]

        master_nodes = master_chain.node_indices
        slave_nodes = slave_chain.node_indices

        # Add directed edges: master → slave (periodic)
        for m_node in master_nodes:
            for s_node in slave_nodes:
                all_master_idx.append(m_node)
                all_slave_idx.append(s_node)
                # Periodic edge (not anti-periodic for now, default to +1.0)
                all_attrs.append(1.0 if not prior.anti_periodic_prior else -1.0)

        # Add reverse edges: slave → master (bidirectional)
        for s_node in slave_nodes:
            for m_node in master_nodes:
                all_master_idx.append(s_node)
                all_slave_idx.append(m_node)
                all_attrs.append(1.0 if not prior.anti_periodic_prior else -1.0)

    if not all_master_idx:
        return None, "E-PBC-PAIR-001"

    pbc_edge_index = np.array(
        [all_master_idx, all_slave_idx], dtype=np.int64
    )
    pbc_edge_attr = np.array(all_attrs, dtype=np.float32).reshape(-1, 1)

    pair_set = PeriodicPairSet(
        pairs=tuple(candidate_pairs),
        pbc_edge_index=pbc_edge_index,
        pbc_edge_attr=pbc_edge_attr,
    )

    return pair_set, None
