"""Tests for phase1_static.pbc_prior global prior inference."""

import numpy as np
import pytest

from phase1_static.pbc_boundary import extract_boundary_chains_from_mesh
from phase1_static.pbc_prior import infer_global_prior


@pytest.fixture
def annular_sector_mesh():
    """Synthetic 6-node annular sector (same as in boundary tests)."""
    angles = np.array([0.0, 22.5, 45.0]) * np.pi / 180.0
    inner_nodes = np.column_stack([np.cos(angles), np.sin(angles)])
    outer_nodes = 2.0 * np.column_stack([np.cos(angles), np.sin(angles)])
    nodes_xy = np.vstack([inner_nodes, outer_nodes]).astype(np.float64)

    triangles = np.array(
        [[0, 1, 4], [0, 4, 3], [1, 2, 5], [1, 5, 4]], dtype=np.int32
    )
    region_code = np.array([1, 1, 2, 2], dtype=np.int32)

    return nodes_xy, triangles, region_code


def test_infer_global_prior_from_sector_mesh(annular_sector_mesh):
    """Test prior inference from boundary chain analysis."""
    nodes_xy, triangles, region_code = annular_sector_mesh

    # Extract chains
    chain_set, err = extract_boundary_chains_from_mesh(nodes_xy, triangles, region_code)
    assert err is None and chain_set is not None

    # Infer prior
    prior = infer_global_prior(chain_set)

    # Verify structure
    assert isinstance(prior.n_geom_prior, int)
    assert prior.n_geom_prior > 0
    assert isinstance(prior.anti_periodic_prior, bool)
    assert isinstance(prior.evidence_notes, str)
    print(f"n_geom_prior={prior.n_geom_prior}, anti_periodic={prior.anti_periodic_prior}")
    print(f"Evidence: {prior.evidence_notes}")


def test_infer_global_prior_anti_periodic_parity():
    """Test that anti_periodic_prior correlates with n_geom parity."""
    # Create a minimal mock chain set
    from phase1_static.pbc_contracts import BoundaryChain, BoundaryChainSet

    # 2 radial chains
    chain1 = BoundaryChain(
        node_indices=np.array([0, 1], dtype=np.int32),
        chain_type="radial",
        endpoint_pair=(0, 1),
    )
    chain2 = BoundaryChain(
        node_indices=np.array([2, 3], dtype=np.int32),
        chain_type="radial",
        endpoint_pair=(2, 3),
    )

    chain_set = BoundaryChainSet(
        chains=(chain1, chain2),
        rotation_origin_xy=np.array([0.0, 0.0], dtype=np.float64),
        mesh_hash="test",
    )

    prior = infer_global_prior(chain_set)

    # 2 radials → 8-fold (even) → anti_periodic should be True
    assert prior.n_geom_prior == 8
    assert prior.anti_periodic_prior is True


def test_infer_global_prior_no_radials():
    """Test prior inference with no radial chains."""
    from phase1_static.pbc_contracts import BoundaryChain, BoundaryChainSet

    # Only arc chains
    chain = BoundaryChain(
        node_indices=np.array([0, 1, 2], dtype=np.int32),
        chain_type="arc",
        endpoint_pair=(0, 2),
    )

    chain_set = BoundaryChainSet(
        chains=(chain,),
        rotation_origin_xy=np.array([0.0, 0.0], dtype=np.float64),
        mesh_hash="test",
    )

    prior = infer_global_prior(chain_set)

    # No radial chains → default to 8-fold
    assert prior.n_geom_prior == 8
    assert "No radial" in prior.evidence_notes
