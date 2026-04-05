"""Immutable temporal data objects for Phase 2 dynamic pipeline.

Contract:
  - All dataclasses are frozen=True (immutable after construction)
  - Channel order y: [Bx, By, A, J] — frozen, never reorder
  - pos is shared/static across all timesteps for a given case
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TemporalMotorSample:
    """Immutable per-case temporal motor sample.

    Shape conventions (T = sequence_len, N = nodes, E = edges):
        pos:                    [N, 2]      float32  (mm, static)
        static_edge_index:      [2, E_static]  int64
        dynamic_edge_index:     [T, 2, E_dyn]  int64
        edge_attr:              [T, E_total, F_edge]  float32
        x:                      [T, N, F_node]  float32
        y:                      [T, N, 4]   float32  channels: [Bx, By, A, J]
        rotor_angles_deg:       [T]          float64
        case_id:                str
        sequence_len:           int (== T)
    """

    pos: np.ndarray
    static_edge_index: np.ndarray
    dynamic_edge_index: np.ndarray
    edge_attr: np.ndarray
    x: np.ndarray
    y: np.ndarray
    rotor_angles_deg: np.ndarray
    case_id: str
    sequence_len: int

    # Frozen channel order contract
    CHANNEL_ORDER: tuple = ("Bx", "By", "A", "J")

    def __post_init__(self):
        # Shape invariant checks (run once at construction)
        T = self.sequence_len
        if self.y.shape[-1] != 4:
            raise ValueError(f"y last dim must be 4 (got {self.y.shape[-1]}). Channel order [Bx,By,A,J] is frozen.")
        if self.y.shape[0] != T:
            raise ValueError(f"y.shape[0]={self.y.shape[0]} != sequence_len={T}")
        if self.x.shape[0] != T:
            raise ValueError(f"x.shape[0]={self.x.shape[0]} != sequence_len={T}")
        if self.rotor_angles_deg.shape[0] != T:
            raise ValueError(f"rotor_angles_deg length {self.rotor_angles_deg.shape[0]} != sequence_len={T}")


@dataclass(frozen=True)
class NormStats:
    """Frozen per-feature normalization statistics.

    Compute once on training split, persist to results/norm_stats.json.
    Apply identically to val/test — do NOT refit on val/test.
    """

    mean: np.ndarray   # shape [C]
    std: np.ndarray    # shape [C]
    epsilon: float = 1e-8

    def __post_init__(self):
        if self.mean.shape != self.std.shape:
            raise ValueError("mean and std must have the same shape")
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive")


def normalize(x: np.ndarray, stats: NormStats) -> np.ndarray:
    """Pure normalization: (x - mean) / (std + epsilon).

    No in-place mutation. Returns new array.
    """
    return (x - stats.mean) / (stats.std + stats.epsilon)


def denormalize(x_norm: np.ndarray, stats: NormStats) -> np.ndarray:
    """Inverse of normalize. Pure function."""
    return x_norm * (stats.std + stats.epsilon) + stats.mean
