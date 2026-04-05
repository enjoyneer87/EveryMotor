"""phase2_dynamic: Dynamic sliding-band time-series pipeline for motor GNN.

Phase 2 adds per-timestep edge topology refresh while reusing static region graphs.
See .github/plans/phase2_dynamic_edge_pipeline.md for the full contract.

Public API:
    TemporalMotorSample    — immutable per-case temporal data object
    NormStats              — frozen per-feature normalization statistics
    build_sliding_band_edges — pure function: rotor-angle-aware gap edge builder
    collate_temporal_batch   — torch-free collation helper for DataLoader
"""

from phase2_dynamic.temporal_sample import TemporalMotorSample, NormStats, normalize
from phase2_dynamic.sliding_band import build_sliding_band_edges
from phase2_dynamic.collate import collate_temporal_batch

__all__ = [
    "TemporalMotorSample",
    "NormStats",
    "normalize",
    "build_sliding_band_edges",
    "collate_temporal_batch",
]
