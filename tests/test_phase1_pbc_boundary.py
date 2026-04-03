"""Tests for phase1_static.pbc_boundary chain extraction."""

import numpy as np
import pytest

from phase1_static.pbc_boundary import (
    compute_chain_descriptor,
    extract_boundary_chains,
    extract_boundary_chains_from_mesh,
    find_region_interface_edges,
)
from phase1_static.pbc_contracts import (
    CanonicalMeshObservation,
    FrozenMapping,
)


@pytest.fixture
def annular_sector_mesh():
    """Synthetic 6-node annular sector for testing.

    Geometry:
      Inner ring (r=1): nodes 0,1,2 at angles 0°, 22.5°, 45°
      Outer ring (r=2): nodes 3,4,5 at same angles

    Triangles (CCW):
      tri0: (0,1,4) region=1    (inner-to-outer sector)
      tri1: (0,4,3) region=1
      tri2: (1,2,5) region=2
      tri3: (1,5,4) region=2

    Expected boundary chains (after split at corners):
      chain0: (0,1,2) → arc (outer arc)
      chain1: (2,5) → radial (outer-to-inner at 45°)
      chain2: (5,4,3) → arc (inner arc, reverse)
      chain3: (3,0) → radial (inner-to-outer at 0°)

    Chain types: [arc, radial, arc, radial]
    Expected origin (approximate): (0, 0) since geometry is concentric
    """
    angles = np.array([0.0, 22.5, 45.0]) * np.pi / 180.0

    inner_nodes = np.column_stack([np.cos(angles), np.sin(angles)])
    outer_nodes = 2.0 * np.column_stack([np.cos(angles), np.sin(angles)])

    nodes_xy = np.vstack([inner_nodes, outer_nodes]).astype(np.float64)
    # nodes: [0,1,2] = inner, [3,4,5] = outer

    triangles = np.array(
        [[0, 1, 4], [0, 4, 3], [1, 2, 5], [1, 5, 4]], dtype=np.int32
    )
    region_code = np.array([1, 1, 2, 2], dtype=np.int32)

    return nodes_xy, triangles, region_code


def test_extract_boundary_chains_is_deterministic_for_sector_mesh(annular_sector_mesh):
    """Test that boundary extraction is deterministic for sector mesh."""
    nodes_xy, triangles, region_code = annular_sector_mesh

    # Run extraction (deterministic)
    chain_set_1, err_1 = extract_boundary_chains_from_mesh(nodes_xy, triangles, region_code)
    chain_set_2, err_2 = extract_boundary_chains_from_mesh(nodes_xy, triangles, region_code)

    assert err_1 is None and err_2 is None
    assert chain_set_1 is not None and chain_set_2 is not None

    # Check determinism: same number of chains
    assert len(chain_set_1.chains) == len(chain_set_2.chains)
    assert len(chain_set_1.chains) > 0, "At least one chain expected"

    # Check origin is close (within numerical precision)
    origin_1 = chain_set_1.rotation_origin_xy
    origin_2 = chain_set_2.rotation_origin_xy
    np.testing.assert_allclose(origin_1, origin_2, atol=1e-10)

    # Check mesh hash is deterministic
    assert chain_set_1.mesh_hash == chain_set_2.mesh_hash


def test_find_region_interface_edges_detects_region_transition(annular_sector_mesh):
    """Test detection of edges between different region codes."""
    nodes_xy, triangles, region_code = annular_sector_mesh

    interface = find_region_interface_edges(triangles, region_code)

    # Interface should include edge (1,4) and (1,5) separating region 1 and 2
    # Topologically: (1,4) appears in tri1 (region 1) and tri3 (region 2)
    #                (1,5) appears in tri2 (region 2) and tri3 (region 2)
    #                So (1,5) is NOT interface (same region)
    #                (1,4) IS interface

    interface_list = sorted([tuple(sorted(e)) for e in interface])
    print(f"Interface edges: {interface_list}")

    # At minimum, edge (1,4) should be an interface
    assert (1, 4) in interface


def test_extract_boundary_chains_returns_error_for_empty_boundary():
    """Test error handling for empty mesh."""
    nodes_xy = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64)
    triangles = np.empty((0, 3), dtype=np.int32)
    region_code = np.empty(0, dtype=np.int32)

    chain_set, error_code = extract_boundary_chains_from_mesh(
        nodes_xy, triangles, region_code
    )

    assert chain_set is None
    assert error_code == "E-MESH-BOUNDARY-001"


def test_extract_boundary_chains_with_canonical_observation(annular_sector_mesh):
    """Test extraction via CanonicalMeshObservation wrapper."""
    nodes_xy, triangles, region_code = annular_sector_mesh

    obs = CanonicalMeshObservation(
        nodes_xy=nodes_xy,
        triangles=triangles,
        region_code=region_code,
        field_dict=FrozenMapping({}),
        sector_id="1/8",
    )

    chain_set, error_code = extract_boundary_chains(obs)

    assert error_code is None
    assert chain_set is not None
    assert len(chain_set.chains) > 0  # At least some chains extracted


def test_chain_descriptor_arc_vs_radial():
    """Test chain type classification."""
    # Arc: constant radius, varying angle
    theta = np.linspace(0, np.pi / 4, 5)
    arc_xy = np.column_stack([np.cos(theta), np.sin(theta)])
    origin = np.array([0.0, 0.0])

    arc_type = compute_chain_descriptor(arc_xy, origin)
    assert arc_type == "arc", f"Expected 'arc', got '{arc_type}'"

    # Radial: varying radius, constant angle (pointing in one direction)
    radii = np.linspace(1.0, 2.0, 5)
    radial_xy = np.column_stack([radii, np.zeros(5)])
    radial_type = compute_chain_descriptor(radial_xy, origin)
    assert radial_type == "radial", f"Expected 'radial', got '{radial_type}'"
