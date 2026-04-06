"""Tests for PBC integration in phase1_static.motor_dataset."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

from phase1_static.motor_dataset import (
    build_samples_from_npz,
    build_sector_pbc_edges_from_mesh,
)


def _build_radial_strip(
    radii: np.ndarray,
    theta_deg: tuple[float, float],
    *,
    reg_code: int,
    start_idx: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    theta = np.deg2rad(np.asarray(theta_deg, dtype=np.float64))
    nodes = []
    for radius in radii.tolist():
        for angle in theta.tolist():
            nodes.append(
                [radius * np.cos(angle), radius * np.sin(angle)]
            )

    triangles = []
    reg_codes = []
    for ridx in range(len(radii) - 1):
        left0 = start_idx + 2 * ridx
        right0 = start_idx + 2 * ridx + 1
        left1 = start_idx + 2 * (ridx + 1)
        right1 = start_idx + 2 * (ridx + 1) + 1
        triangles.append([left0, right0, right1])
        triangles.append([left0, right1, left1])
        reg_codes.extend([reg_code, reg_code])

    return (
        np.asarray(nodes, dtype=np.float32),
        np.asarray(triangles, dtype=np.int32),
        np.asarray(reg_codes, dtype=np.int32),
    )


def test_build_sector_pbc_edges_from_mesh_matches_45deg_sector() -> None:
    """Verify deterministic 45-degree PBC matching on synthetic annular sector."""
    angles = np.deg2rad(np.array([0.0, 22.5, 45.0], dtype=np.float64))
    inner = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    outer = 2.0 * inner
    pos = np.vstack([inner, outer]).astype(np.float32)

    triangles = np.array(
        [
            [0, 1, 4],
            [0, 4, 3],
            [1, 2, 5],
            [1, 5, 4],
        ],
        dtype=np.int32,
    )

    pbc_edge_index, pbc_edge_attr, diag = build_sector_pbc_edges_from_mesh(
        pos_xy=pos,
        triangles=triangles,
        rotation_deg=-45.0,
        anti_periodic=True,
    )

    assert diag["error_code"] == ""
    assert pbc_edge_index.shape[0] == 2
    assert pbc_edge_attr.shape[0] == pbc_edge_index.shape[1]
    assert np.allclose(pbc_edge_attr, -1.0)
    assert float(diag["match_ratio"]) > 0.99
    assert float(diag["max_rotation_residual"]) < 1e-3


def test_build_sector_pbc_edges_from_mesh_tracks_offset_rotor_and_stator() -> None:
    """Rotor and stator cuts should both be matched when reg metadata is available."""
    rotor_pos, rotor_tri, rotor_reg = _build_radial_strip(
        np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
        (-65.0, -20.0),
        reg_code=10,
        start_idx=0,
    )
    stator_pos, stator_tri, stator_reg = _build_radial_strip(
        np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float64),
        (-45.0, 0.0),
        reg_code=1,
        start_idx=rotor_pos.shape[0],
    )

    pos = np.vstack([rotor_pos, stator_pos]).astype(np.float32)
    triangles = np.vstack([rotor_tri, stator_tri]).astype(np.int32)
    region_code = np.concatenate([rotor_reg, stator_reg]).astype(np.int32)

    pbc_edge_index, pbc_edge_attr, diag = build_sector_pbc_edges_from_mesh(
        pos_xy=pos,
        triangles=triangles,
        region_code=region_code,
        moving_reg_codes=np.asarray([10], dtype=np.int32),
        region_name_by_code={1: "Stator", 10: "Rotor"},
        rotation_deg=-45.0,
        anti_periodic=True,
    )

    assert diag["error_code"] == ""
    assert int(diag["group_count"]) == 2
    assert pbc_edge_index.shape == (2, 16)
    assert pbc_edge_attr.shape == (16, 1)
    assert np.allclose(pbc_edge_attr, -1.0)
    assert float(diag["match_ratio"]) > 0.99


def test_build_samples_from_npz_preserves_precomputed_pbc(tmp_path) -> None:
    """When NPZ already has PBC edges, loader should keep them unchanged."""
    npz_path = tmp_path / "bundle.npz"

    pos = np.array([[[0.0, 0.0], [1.0, 0.0]]], dtype=np.float32)
    node_type = np.array([[[1.0], [1.0]]], dtype=np.float32)
    interior = np.array([[[0, 1], [1, 0]]], dtype=np.int64)
    pbc_edge_index = np.array([[[0, 1], [1, 0]]], dtype=np.int64)
    pbc_edge_attr = np.array([[[-1.0], [-1.0]]], dtype=np.float32)
    y = np.array([[[0.1, 0.2, 0.0, 0.0], [0.2, 0.3, 0.0, 0.0]]], dtype=np.float32)

    np.savez(
        npz_path,
        pos=pos,
        node_type_onehot=node_type,
        interior_edge_index=interior,
        pbc_edge_index=pbc_edge_index,
        pbc_edge_attr=pbc_edge_attr,
        y=y,
    )

    samples = build_samples_from_npz(str(npz_path))
    assert len(samples) == 1
    np.testing.assert_array_equal(samples[0]["pbc_edge_index"], pbc_edge_index[0])
    np.testing.assert_array_equal(samples[0]["pbc_edge_attr"], pbc_edge_attr[0])
