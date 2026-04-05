"""Tests for PBC integration in phase1_static.motor_dataset."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

from phase1_static.motor_dataset import (
    build_samples_from_npz,
    build_sector_pbc_edges_from_mesh,
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
