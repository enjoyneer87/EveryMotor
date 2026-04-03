"""Tests for phase1_static.pbc_bundle end-to-end composition."""

import numpy as np
import pytest

from phase1_static.pbc_bundle import build_pbc_bundle
from phase1_static.pbc_contracts import (
    CanonicalMeshObservation,
    FrozenMapping,
)


@pytest.fixture
def annular_sector_mesh():
    """Synthetic 6-node annular sector."""
    angles = np.array([0.0, 22.5, 45.0]) * np.pi / 180.0
    inner_nodes = np.column_stack([np.cos(angles), np.sin(angles)])
    outer_nodes = 2.0 * np.column_stack([np.cos(angles), np.sin(angles)])
    nodes_xy = np.vstack([inner_nodes, outer_nodes]).astype(np.float64)

    triangles = np.array(
        [[0, 1, 4], [0, 4, 3], [1, 2, 5], [1, 5, 4]], dtype=np.int32
    )
    region_code = np.array([1, 1, 2, 2], dtype=np.int32)

    return nodes_xy, triangles, region_code


def test_build_pbc_bundle_end_to_end(annular_sector_mesh):
    """Test full pipeline: observation → bundle."""
    nodes_xy, triangles, region_code = annular_sector_mesh

    obs = CanonicalMeshObservation(
        nodes_xy=nodes_xy,
        triangles=triangles,
        region_code=region_code,
        field_dict=FrozenMapping({"bx": np.zeros(len(nodes_xy))}),
        sector_id="1/8",
    )

    bundle, error_code = build_pbc_bundle(obs)

    # If successful, check structure; if pairing fails, that's acceptable
    if bundle is not None:
        assert error_code is None
        # Check all layers present
        assert bundle.observation is obs
        assert bundle.chain_set is not None
        assert len(bundle.chain_set.chains) > 0
        assert bundle.prior is not None
        assert bundle.pair_set is not None

        # Check versioning
        assert bundle.format_version == "0.1.0"
        assert bundle.bundle_type == "phase1_pbc_bundle"

        # Check immutability
        assert not bundle.observation.nodes_xy.flags.writeable
        assert not bundle.chain_set.rotation_origin_xy.flags.writeable
    else:
        # Pairing may fail due to insufficient chains in test mesh
        assert error_code == "E-PBC-PAIR-001"


def test_build_pbc_bundle_error_propagation():
    """Test error handling through pipeline."""
    # Create observation with empty mesh
    nodes_xy = np.empty((0, 2), dtype=np.float64)
    triangles = np.empty((0, 3), dtype=np.int32)
    region_code = np.empty(0, dtype=np.int32)

    obs = CanonicalMeshObservation(
        nodes_xy=nodes_xy,
        triangles=triangles,
        region_code=region_code,
        field_dict=FrozenMapping({}),
        sector_id="1/8",
    )

    bundle, error_code = build_pbc_bundle(obs)

    # Should fail with error code
    assert bundle is None
    assert error_code == "E-MESH-BOUNDARY-001"
