"""Band spectral loss plumbing: batching offsets and segmented coefficients.

The failure mode this guards is silent: PyG's default `__inc__` shifts any key
containing "index" by `num_nodes`, but `band_row_index` points into the
OPERATOR ROWS (`b_true`), so without the `CurlData` override every batched
graph's band would read the first graph's elements — training would run,
converge, and quietly optimise the wrong thing.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

import torch  # noqa: E402
from torch_geometric.loader import DataLoader  # noqa: E402

from train_doe_curl_mgn import (  # noqa: E402
    BAND_SPECTRAL_ORDERS,
    CurlData,
    band_coefficients,
)


def _graph(n_nodes: int, n_rows: int, band_rows: np.ndarray, ext: np.ndarray) -> CurlData:
    """Minimal CurlData carrying just what batching and the loss touch."""
    n_band = band_rows.size
    phi = np.linspace(0.1, 0.6, n_band)
    return CurlData(
        x=torch.zeros((n_nodes, 3)),
        edge_index=torch.zeros((2, 1), dtype=torch.long),
        b_true=torch.randn(n_rows, 2),
        band_row_index=torch.from_numpy(band_rows.astype(np.int64)),
        band_cos=torch.from_numpy(np.cos(phi).astype(np.float32)),
        band_sin=torch.from_numpy(np.sin(phi).astype(np.float32)),
        band_ext=torch.from_numpy(ext.astype(np.float32)),
        band_count=torch.tensor([n_band], dtype=torch.long),
        num_nodes=n_nodes,
    )


def test_band_row_index_offsets_by_operator_rows_not_nodes():
    ncoef = 6
    g1 = _graph(n_nodes=10, n_rows=7, band_rows=np.array([1, 3, 5]),
                ext=np.random.rand(3, ncoef))
    g2 = _graph(n_nodes=4, n_rows=9, band_rows=np.array([0, 2, 4, 8]),
                ext=np.random.rand(4, ncoef))
    batch = next(iter(DataLoader([g1, g2], batch_size=2)))

    # Graph 2's rows must be shifted by graph 1's OPERATOR row count (7), not by
    # its node count (10).
    expected = np.concatenate([[1, 3, 5], np.array([0, 2, 4, 8]) + 7])
    assert batch.band_row_index.tolist() == expected.tolist()
    assert batch.band_count.tolist() == [3, 4]
    assert batch.b_true.shape[0] == 16


def test_segmented_coefficients_match_per_graph_matmul():
    rng = np.random.default_rng(0)
    ncoef = 5
    ext1, ext2 = rng.random((3, ncoef)), rng.random((4, ncoef))
    g1 = _graph(10, 7, np.array([1, 3, 5]), ext1)
    g2 = _graph(4, 9, np.array([0, 2, 4, 8]), ext2)
    batch = next(iter(DataLoader([g1, g2], batch_size=2)))

    values = torch.arange(7, dtype=torch.float32)  # one scalar per band row
    out = band_coefficients(batch, values)

    ref1 = ext1.T @ values[:3].numpy()
    ref2 = ext2.T @ values[3:].numpy()
    np.testing.assert_allclose(out[0].numpy(), ref1, rtol=1e-5)
    np.testing.assert_allclose(out[1].numpy(), ref2, rtol=1e-5)


def test_extraction_matrix_round_trips_known_coefficients():
    """ext = pinv(design).T must recover the coefficients of a synthetic field."""
    rng = np.random.default_rng(1)
    phi = np.deg2rad(np.linspace(-44.0, -1.0, 300))  # a 45-deg arc, like the band
    cols = []
    for k in BAND_SPECTRAL_ORDERS:
        cols.append(np.cos(k * phi))
        cols.append(np.sin(k * phi))
    design = np.column_stack(cols)
    ext = np.linalg.pinv(design).T

    c = rng.standard_normal(design.shape[1])
    field = design @ c
    recovered = ext.T @ field
    np.testing.assert_allclose(recovered, c, atol=1e-8)


def test_graph_without_band_contributes_empty_rows():
    ncoef = 2 * len(BAND_SPECTRAL_ORDERS)
    g1 = _graph(10, 7, np.array([1, 3, 5]), np.random.rand(3, ncoef))
    g2 = _graph(4, 9, np.zeros(0, dtype=np.int64), np.zeros((0, ncoef)))
    batch = next(iter(DataLoader([g1, g2], batch_size=2)))
    values = torch.ones(3)
    out = band_coefficients(batch, values)
    assert out.shape == (2, ncoef)
    assert torch.all(out[1] == 0)  # the bandless graph stays zero, no bleed-over
