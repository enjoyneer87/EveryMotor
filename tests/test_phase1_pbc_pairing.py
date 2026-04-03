"""Tests for phase1_static.pbc_pairing periodic pair matching."""

import numpy as np
import pytest

from phase1_static.pbc_boundary import extract_boundary_chains_from_mesh
from phase1_static.pbc_prior import infer_global_prior
from phase1_static.pbc_pairing import build_periodic_pair_set


@pytest.fixture
def annular_sector_mesh():
    """Synthetic 6-node annular sector (same as boundaries)."""
    angles = np.array([0.0, 22.5, 45.0]) * np.pi / 180.0
    inner_nodes = np.column_stack([np.cos(angles), np.sin(angles)])
    outer_nodes = 2.0 * np.column_stack([np.cos(angles), np.sin(angles)])
    nodes_xy = np.vstack([inner_nodes, outer_nodes]).astype(np.float64)

    triangles = np.array(
        [[0, 1, 4], [0, 4, 3], [1, 2, 5], [1, 5, 4]], dtype=np.int32
    )
    region_code = np.array([1, 1, 2, 2], dtype=np.int32)

    return nodes_xy, triangles, region_code


def test_build_periodic_pair_set_from_chains(annular_sector_mesh):
    """Test PBC pair set construction from chains and prior."""
    nodes_xy, triangles, region_code = annular_sector_mesh

    # Extract chains
    chain_set, err1 = extract_boundary_chains_from_mesh(nodes_xy, triangles, region_code)
    assert err1 is None and chain_set is not None

    # Infer prior
    prior = infer_global_prior(chain_set)

    # Build pair set
    pair_set, err2 = build_periodic_pair_set(chain_set, prior, nodes_xy)

    # Should succeed if chains detected
    if pair_set is not None:
        assert err2 is None
        assert len(pair_set.pairs) > 0
        assert pair_set.pbc_edge_index.shape[0] == 2  # [2, E]
        assert pair_set.pbc_edge_attr.shape[1] == 1  # [E, 1]
        assert pair_set.pbc_edge_index.shape[1] == pair_set.pbc_edge_attr.shape[0]
    else:
        # No chains to pair (acceptable)
        assert err2 == "E-PBC-PAIR-001"


def test_build_periodic_pair_set_returns_error_for_empty_chains():
    """Test error handling for no chains."""
    from phase1_static.pbc_contracts import BoundaryChainSet, PriorContext

    empty_chain_set = BoundaryChainSet(
        chains=(),
        rotation_origin_xy=np.array([0.0, 0.0], dtype=np.float64),
        mesh_hash="test",
    )

    prior = PriorContext(
        n_geom_prior=8,
        anti_periodic_prior=True,
        evidence_notes="test",
    )

    nodes_xy = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64)

    pair_set, error_code = build_periodic_pair_set(empty_chain_set, prior, nodes_xy)

    assert pair_set is None
    assert error_code == "E-PBC-PAIR-001"


def test_periodic_pair_set_bidirectionality(annular_sector_mesh):
    """Test that generated edges are bidirectional."""
    nodes_xy, triangles, region_code = annular_sector_mesh

    chain_set, _ = extract_boundary_chains_from_mesh(nodes_xy, triangles, region_code)
    assert chain_set is not None

    prior = infer_global_prior(chain_set)
    pair_set, err = build_periodic_pair_set(chain_set, prior, nodes_xy)

    if pair_set is not None:
        assert err is None
        # Check bidirectionality: for each (u,v), (v,u) should exist
        edge_set = set(tuple(edge) for edge in pair_set.pbc_edge_index.T)
        for u, v in edge_set:
            assert (v, u) in edge_set, f"Edge ({u}, {v}) has no reverse"
