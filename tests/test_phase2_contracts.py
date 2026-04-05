"""Phase 2 contract tests — torch-free, runs on host venv.

Tests:
  - TemporalMotorSample shape invariants
  - NormStats normalize/denormalize roundtrip
  - build_sliding_band_edges returns [2,E] and no duplicates
  - collate_temporal_batch stacks correctly
"""

from __future__ import annotations

import numpy as np
import pytest

from phase2_dynamic.temporal_sample import (
    TemporalMotorSample,
    NormStats,
    normalize,
    denormalize,
)
from phase2_dynamic.sliding_band import build_sliding_band_edges
from phase2_dynamic.collate import collate_temporal_batch


# ─── helpers ────────────────────────────────────────────────────────────────

def _make_sample(T=3, N=10, E_static=20, E_dyn=8, F_node=6, F_edge=1) -> TemporalMotorSample:
    rng = np.random.default_rng(0)
    return TemporalMotorSample(
        pos=rng.random((N, 2)).astype(np.float32),
        static_edge_index=rng.integers(0, N, size=(2, E_static)).astype(np.int64),
        dynamic_edge_index=rng.integers(0, N, size=(T, 2, E_dyn)).astype(np.int64),
        edge_attr=rng.random((T, E_static + E_dyn, F_edge)).astype(np.float32),
        x=rng.random((T, N, F_node)).astype(np.float32),
        y=rng.random((T, N, 4)).astype(np.float32),
        rotor_angles_deg=np.linspace(0.0, 45.0, T),
        case_id="test_case_0001",
        sequence_len=T,
    )


# ─── TemporalMotorSample ────────────────────────────────────────────────────

def test_temporal_sample_construction():
    s = _make_sample()
    assert s.y.shape == (3, 10, 4)
    assert s.x.shape == (3, 10, 6)
    assert s.sequence_len == 3


def test_temporal_sample_channel_order_frozen():
    """y must have exactly 4 output channels."""
    rng = np.random.default_rng(1)
    with pytest.raises(ValueError, match="4"):
        TemporalMotorSample(
            pos=rng.random((5, 2)).astype(np.float32),
            static_edge_index=np.zeros((2, 4), dtype=np.int64),
            dynamic_edge_index=np.zeros((2, 2, 4), dtype=np.int64),
            edge_attr=np.zeros((2, 8, 1), dtype=np.float32),
            x=rng.random((2, 5, 3)).astype(np.float32),
            y=rng.random((2, 5, 3)).astype(np.float32),  # wrong: 3 not 4
            rotor_angles_deg=np.array([0.0, 10.0]),
            case_id="bad",
            sequence_len=2,
        )


def test_temporal_sample_sequence_len_mismatch():
    rng = np.random.default_rng(2)
    with pytest.raises(ValueError, match="sequence_len"):
        TemporalMotorSample(
            pos=rng.random((5, 2)).astype(np.float32),
            static_edge_index=np.zeros((2, 4), dtype=np.int64),
            dynamic_edge_index=np.zeros((2, 2, 4), dtype=np.int64),
            edge_attr=np.zeros((2, 8, 1), dtype=np.float32),
            x=rng.random((2, 5, 3)).astype(np.float32),
            y=rng.random((3, 5, 4)).astype(np.float32),  # T=3 but sequence_len=2
            rotor_angles_deg=np.array([0.0, 10.0]),
            case_id="bad",
            sequence_len=2,
        )


# ─── NormStats ──────────────────────────────────────────────────────────────

def test_normalize_denormalize_roundtrip():
    rng = np.random.default_rng(3)
    x = rng.random((10, 4)).astype(np.float32)
    stats = NormStats(mean=x.mean(0), std=x.std(0) + 0.1)
    x_norm = normalize(x, stats)
    x_rec = denormalize(x_norm, stats)
    np.testing.assert_allclose(x_rec, x, rtol=1e-5, atol=1e-6)


def test_normalize_is_pure():
    """normalize must not mutate the input."""
    rng = np.random.default_rng(4)
    x = rng.random((5, 4)).astype(np.float32)
    x_copy = x.copy()
    stats = NormStats(mean=np.zeros(4, dtype=np.float32), std=np.ones(4, dtype=np.float32))
    _ = normalize(x, stats)
    np.testing.assert_array_equal(x, x_copy)


# ─── build_sliding_band_edges ───────────────────────────────────────────────

def _ring_sector_pos(n=20):
    """Simple ring of nodes for testing gap edge builder."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    inner = np.stack([np.cos(angles), np.sin(angles)], axis=1).astype(np.float32)
    outer = 1.5 * inner
    return np.vstack([inner, outer])


def test_sliding_band_edges_shape():
    pos = _ring_sector_pos(20)
    N = pos.shape[0]
    gap_mask = np.ones(N, dtype=bool)
    stator_idx = np.arange(0, 10, dtype=np.int64)
    rotor_idx = np.arange(20, 30, dtype=np.int64)
    ei, ea = build_sliding_band_edges(
        pos, gap_mask, rotor_angle_deg=0.0,
        stator_indices=stator_idx, rotor_indices=rotor_idx,
        k_neighbors=2, max_dist_mm=5.0,
    )
    assert ei.ndim == 2
    assert ei.shape[0] == 2
    assert ea.shape == (ei.shape[1], 1)


def test_sliding_band_edges_no_duplicates():
    pos = _ring_sector_pos(20)
    N = pos.shape[0]
    gap_mask = np.ones(N, dtype=bool)
    stator_idx = np.arange(0, 10, dtype=np.int64)
    rotor_idx = np.arange(20, 30, dtype=np.int64)
    ei, _ = build_sliding_band_edges(
        pos, gap_mask, rotor_angle_deg=10.0,
        stator_indices=stator_idx, rotor_indices=rotor_idx,
        k_neighbors=3, max_dist_mm=5.0,
    )
    if ei.shape[1] > 0:
        pairs = set(zip(ei[0].tolist(), ei[1].tolist()))
        assert len(pairs) == ei.shape[1], "Duplicate edges found"


def test_sliding_band_edges_empty_input():
    pos = _ring_sector_pos(10)
    gap_mask = np.ones(10, dtype=bool)
    ei, ea = build_sliding_band_edges(
        pos, gap_mask, rotor_angle_deg=0.0,
        stator_indices=np.array([], dtype=np.int64),
        rotor_indices=np.array([], dtype=np.int64),
    )
    assert ei.shape == (2, 0)
    assert ea.shape == (0, 1)


# ─── collate_temporal_batch ─────────────────────────────────────────────────

def test_collate_stacks_correctly():
    samples = [_make_sample(T=3, N=10) for _ in range(4)]
    batch = collate_temporal_batch(samples)
    assert batch["pos"].shape == (4, 10, 2)
    assert batch["y"].shape == (4, 3, 10, 4)
    assert batch["x"].shape == (4, 3, 10, 6)
    assert batch["sequence_len"] == 3
    assert len(batch["case_ids"]) == 4


def test_collate_raises_on_sequence_len_mismatch():
    s1 = _make_sample(T=3)
    s2 = _make_sample(T=4)
    with pytest.raises(ValueError, match="sequence_len"):
        collate_temporal_batch([s1, s2])


def test_collate_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        collate_temporal_batch([])
