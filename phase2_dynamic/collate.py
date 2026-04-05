"""Collation utilities for TemporalMotorSample batches.

Torch-free: returns plain numpy dicts compatible with DataLoader
and convertible to torch tensors in the training loop.
"""

from __future__ import annotations

from typing import List

import numpy as np

from phase2_dynamic.temporal_sample import TemporalMotorSample


def collate_temporal_batch(samples: List[TemporalMotorSample]) -> dict:
    """Collate a list of TemporalMotorSample into a batch dict.

    All samples in a batch must have:
      - same sequence_len T
      - same number of nodes N
      - same static E_static (for simple stack; a sparse variant is future work)

    Returns dict with stacked numpy arrays:
        pos:                [B, N, 2]
        static_edge_index:  [B, 2, E_static]
        dynamic_edge_index: [B, T, 2, E_dyn]
        edge_attr:          [B, T, E_total, F_edge]
        x:                  [B, T, N, F_node]
        y:                  [B, T, N, 4]
        rotor_angles_deg:   [B, T]
        case_ids:           list[str]
        sequence_len:       int
    """
    if not samples:
        raise ValueError("Cannot collate empty list")

    T = samples[0].sequence_len
    for s in samples:
        if s.sequence_len != T:
            raise ValueError(
                f"Inconsistent sequence_len in batch: {s.sequence_len} vs {T}"
            )

    return {
        "pos": np.stack([s.pos for s in samples], axis=0),
        "static_edge_index": np.stack([s.static_edge_index for s in samples], axis=0),
        "dynamic_edge_index": np.stack([s.dynamic_edge_index for s in samples], axis=0),
        "edge_attr": np.stack([s.edge_attr for s in samples], axis=0),
        "x": np.stack([s.x for s in samples], axis=0),
        "y": np.stack([s.y for s in samples], axis=0),
        "rotor_angles_deg": np.stack([s.rotor_angles_deg for s in samples], axis=0),
        "case_ids": [s.case_id for s in samples],
        "sequence_len": T,
    }
