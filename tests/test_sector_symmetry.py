"""Tests for 1/8 sector anti-periodic symmetry handling."""

import json
from pathlib import Path

import numpy as np
import pytest

from phase1_static.sector_symmetry import (
    DEFAULT_SECTOR_DEG,
    SectorSymmetryError,
    anti_periodic_edges,
    anti_periodicity_residual,
    cumulative_rotor_angle,
    cut_plane_pairs,
    rigid_rotor_node_mask,
    rotor_angle_features,
    sector_geometry,
    wrap_rotor_angle,
    wrap_rotor_coordinates,
)

DATA_DIR = Path("backup/doe_data")
MANIFEST = DATA_DIR / "doe_manifest.json"
requires_doe = pytest.mark.skipif(not MANIFEST.exists(), reason="DOE export not present")


def test_sector_divides_the_circle():
    geom = sector_geometry(45.0)
    assert geom.n_sectors == 8
    assert geom.sign_per_sector == -1.0


def test_non_dividing_sector_is_rejected():
    with pytest.raises(SectorSymmetryError, match="does not divide"):
        sector_geometry(37.0)


@pytest.mark.parametrize(
    "cum,expected_wrapped,expected_k,expected_sign",
    [
        (0.0, -45.0, -1, -1.0),
        (-1.0, -1.0, 0, 1.0),
        (-44.0, -44.0, 0, 1.0),
        (-45.0, -45.0, 0, 1.0),
        (-46.0, -1.0, 1, -1.0),
        (-66.0, -21.0, 1, -1.0),
        (-88.0, -43.0, 1, -1.0),
        (-91.0, -1.0, 2, 1.0),
    ],
)
def test_wrap_angle_folds_into_one_sector(cum, expected_wrapped, expected_k, expected_sign):
    wrapped, k, sign = wrap_rotor_angle(cum)
    assert wrapped == pytest.approx(expected_wrapped)
    assert k == expected_k
    assert sign == expected_sign
    assert wrapped == pytest.approx(cum + k * DEFAULT_SECTOR_DEG)


def test_wrapped_angle_always_lands_in_the_sector():
    for cum in np.linspace(-400.0, 10.0, 401):
        wrapped, _, _ = wrap_rotor_angle(float(cum))
        assert -DEFAULT_SECTOR_DEG - 1e-9 <= wrapped < 1e-9


def test_sign_alternates_every_sector():
    assert wrap_rotor_angle(-10.0)[2] == 1.0
    assert wrap_rotor_angle(-55.0)[2] == -1.0
    assert wrap_rotor_angle(-100.0)[2] == 1.0
    assert wrap_rotor_angle(-145.0)[2] == -1.0


def test_cumulative_angle_accumulates_the_increments():
    steps = [0.0, -2.0, -2.0, -2.0, -2.0]
    np.testing.assert_allclose(cumulative_rotor_angle(steps), [0.0, -2.0, -4.0, -6.0, -8.0])


def test_cumulative_angle_tolerates_nan_first_entry():
    np.testing.assert_allclose(cumulative_rotor_angle([np.nan, -2.0, -2.0]), [0.0, -2.0, -4.0])


def test_angle_features_are_antiperiodic_not_periodic():
    """One sector must flip the encoding, two sectors must restore it."""
    base = rotor_angle_features(-10.0)
    one = rotor_angle_features(-10.0 - DEFAULT_SECTOR_DEG)
    two = rotor_angle_features(-10.0 - 2 * DEFAULT_SECTOR_DEG)

    np.testing.assert_allclose(one, [-base[0], -base[1]], atol=1e-12)
    np.testing.assert_allclose(two, base, atol=1e-12)


def test_angle_features_distinguish_steps_unlike_rotate_step():
    """rotate_step is {0,-2} for every step; the encoding must actually vary."""
    values = {rotor_angle_features(-2.0 * i) for i in range(45)}
    assert len(values) == 45


# --- geometry ---------------------------------------------------------------


def _sector_mesh():
    """Two rings of nodes on the theta=0 and theta=-45 cut planes, plus interior."""
    radii = np.array([10.0, 20.0, 30.0])
    xs, ys = [], []
    for th in (0.0, -45.0, -20.0):
        a = np.radians(th)
        xs.append(radii * np.cos(a))
        ys.append(radii * np.sin(a))
    return np.concatenate(xs), np.concatenate(ys)


def test_cut_planes_are_paired_by_radius():
    x, y = _sector_mesh()
    pairs = cut_plane_pairs(x, y)
    assert pairs.shape == (3, 2)

    r = np.hypot(x, y)
    np.testing.assert_allclose(r[pairs[:, 0]], r[pairs[:, 1]], atol=1e-9)
    theta = np.degrees(np.arctan2(y, x))
    np.testing.assert_allclose(theta[pairs[:, 0]], 0.0, atol=1e-6)
    np.testing.assert_allclose(theta[pairs[:, 1]], -45.0, atol=1e-6)


def test_unmatched_radii_are_not_paired():
    x = np.array([10.0, 20.0, 10.0 * np.cos(np.radians(-45))])
    y = np.array([0.0, 0.0, 10.0 * np.sin(np.radians(-45))])
    pairs = cut_plane_pairs(x, y)
    assert pairs.shape == (1, 2)  # only r=10 has a partner


def test_mesh_without_cut_planes_yields_no_pairs():
    x = np.array([1.0, 2.0])
    y = np.array([-1.0, -2.0])
    assert cut_plane_pairs(x, y).shape == (0, 2)


def test_anti_periodic_edges_are_bidirectional_and_negative():
    pairs = np.array([[0, 5], [1, 6]])
    edge_index, sign = anti_periodic_edges(pairs)
    assert edge_index.shape == (2, 4)
    assert np.all(sign == -1.0)
    assert set(map(tuple, edge_index.T.tolist())) == {(0, 5), (1, 6), (5, 0), (6, 1)}


def test_residual_is_zero_for_an_anti_periodic_field():
    pairs = np.array([[0, 1], [2, 3]])
    values = np.array([1.0, -1.0, 2.5, -2.5])
    assert anti_periodicity_residual(values, pairs) == pytest.approx(0.0)


def test_residual_is_two_for_a_periodic_field():
    """A periodic (not anti-periodic) field is maximally wrong."""
    pairs = np.array([[0, 1]])
    assert anti_periodicity_residual(np.array([3.0, 3.0]), pairs) == pytest.approx(2.0)


def test_wrap_is_identity_when_k_is_zero():
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([0.5, -1.0, 2.0])
    mask = np.array([True, True, False])
    wx, wy, sign = wrap_rotor_coordinates(x, y, mask, cumulative_deg=-20.0)
    np.testing.assert_allclose(wx, x)
    np.testing.assert_allclose(wy, y)
    assert sign == 1.0


def test_wrap_rotates_only_the_masked_nodes_and_preserves_radius():
    angle = np.radians(np.array([-60.0, -70.0, -10.0]))
    r = np.array([30.0, 40.0, 50.0])
    x, y = r * np.cos(angle), r * np.sin(angle)
    mask = np.array([True, True, False])

    wx, wy, sign = wrap_rotor_coordinates(x, y, mask, cumulative_deg=-66.0)
    assert sign == -1.0

    np.testing.assert_allclose(np.hypot(wx, wy), r, atol=1e-9)
    np.testing.assert_allclose([wx[2], wy[2]], [x[2], y[2]])  # unmasked untouched
    moved = np.degrees(np.arctan2(wy[:2], wx[:2]))
    np.testing.assert_allclose(moved, np.degrees(angle[:2]) + 45.0, atol=1e-9)


# --- against the real export -------------------------------------------------


@requires_doe
def test_export_is_anti_periodic_not_periodic():
    from eval.doe_dataset import load_doe_cases
    from eval.torque import element_to_nodal

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    record = load_doe_cases(manifest, DATA_DIR, case_indices=[4], max_steps=1).records[0]
    mesh = record.mesh

    pairs = cut_plane_pairs(mesh.node_x_mm, mesh.node_y_mm)
    assert pairs.shape[0] > 50, "expected the cut planes to be matched"

    a = element_to_nodal(record.samples[0].fields["a"], mesh.tri, mesh.n_nodes)
    assert anti_periodicity_residual(a, pairs) < 0.05

    # The periodic hypothesis A(0) == A(-45) must fail badly, otherwise the
    # anti-periodic result above would just mean "A is small on the cut".
    lo, hi = a[pairs[:, 0]], a[pairs[:, 1]]
    periodic = float(np.sqrt(np.mean((lo - hi) ** 2)) / np.sqrt(np.mean(lo**2)))
    assert periodic > 1.5


@requires_doe
def test_rigid_mask_excludes_the_shear_layer():
    from eval.doe_dataset import load_doe_cases

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    mesh = load_doe_cases(manifest, DATA_DIR, case_indices=[4], max_steps=1).records[0].mesh

    mask = rigid_rotor_node_mask(
        mesh.tri, mesh.reg_code, mesh.name_of_code, mesh.moving_reg_codes, mesh.n_nodes
    )
    a2_codes = [c for c, n in mesh.name_of_code.items() if n == "a2"]
    a2_nodes = set()
    for idx in mesh.tri:
        a2_nodes |= set(np.asarray(idx)[np.isin(mesh.reg_code, a2_codes)].tolist())

    assert mask.any()
    assert not any(mask[n] for n in a2_nodes), "a2 nodes must never be rotated"


@requires_doe
def test_wrapping_never_invalidates_a_scoreable_element():
    """Wrapping must not push inversions outside the already-excluded band."""
    from eval.doe_dataset import load_doe_cases
    from eval.mesh_regions import sliding_band_mask
    from phase1_static.discrete_curl import signed_double_area

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    record = load_doe_cases(manifest, DATA_DIR, case_indices=[4], max_steps=45).records[0]
    mesh = record.mesh

    reference = np.sign(signed_double_area(mesh.node_x_mm, mesh.node_y_mm, mesh.tri))
    excluded = sliding_band_mask(mesh.reg_code, mesh.name_of_code, mesh.moving_reg_codes)
    rigid = rigid_rotor_node_mask(
        mesh.tri, mesh.reg_code, mesh.name_of_code, mesh.moving_reg_codes, mesh.n_nodes
    )

    for step in (11, 22, 33, 44):
        sample = record.samples[step]
        x, y, _ = wrap_rotor_coordinates(
            sample.node_x_mm, sample.node_y_mm, rigid, cumulative_deg=-2.0 * step
        )
        inverted = np.sign(signed_double_area(x, y, mesh.tri)) != reference
        assert not np.any(inverted & ~excluded), (
            f"step {step}: wrapping inverted {int((inverted & ~excluded).sum())} scoreable elements"
        )
