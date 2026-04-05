# Phase 2: Dynamic Sliding-Band Time-Series Pipeline — Contract Document

> **Agents:** Read this file AND `AGENTS.md` before editing any Phase 2 code.  
> **Gate:** This document is required reading before opening any PR that touches `phase2_dynamic/`.

---

## Overview

Phase 2 adds time-varying support.  
Motor topology changes per rotor angle: the gap/sliding band edges must be rebuilt every timestep while the static stator/rotor interior topology is reused.

---

## Data Object Contract

### `TemporalMotorSample` (immutable dataclass)

```python
@dataclass(frozen=True)
class TemporalMotorSample:
    # Geometry (static, shared across time)
    pos: np.ndarray           # shape [N, 2], float32, units: mm
    static_edge_index: np.ndarray  # shape [2, E_static], int64

    # Per-step dynamic topology
    dynamic_edge_index: np.ndarray  # shape [T, 2, E_dyn], int64
    edge_attr: np.ndarray           # shape [T, E_total, F_edge], float32

    # Node features per step
    x: np.ndarray             # shape [T, N, F_node], float32

    # Targets per step
    y: np.ndarray             # shape [T, N, 4], float32
                              # channel order: [Bx, By, A, J] — FROZEN, do not change

    # Sequence metadata
    rotor_angles_deg: np.ndarray  # shape [T], float64
    case_id: str
    sequence_len: int         # T
```

**Invariants:**
- `y.shape[-1] == 4` always (channel order `[Bx, By, A, J]`)
- `pos` is the same across all timesteps for a given case
- `x.shape[0] == y.shape[0] == dynamic_edge_index.shape[0] == sequence_len`

---

## Pipeline Stage Contract

```
raw FEA files (per timestep)
    ↓  [TemporalMotorDataset]
TemporalMotorSample  (immutable, per-case)
    ↓  [collate_temporal_batch]
Batch dict {pos, static_edge_index, dynamic_edge_index, x, y, ...}
    ↓  [TemporalMeshGraphNet.forward(x_t, edge_index_t)]
pred_y  shape [T, N, 4]
    ↓  [temporal_hybrid_loss]
scalar loss
```

---

## Sliding-Band Edge Refresh Contract

The dynamic edge region covers nodes within `gap_region_mask == True`.

```python
def build_sliding_band_edges(
    pos: np.ndarray,           # [N, 2]
    gap_mask: np.ndarray,      # [N], bool
    rotor_angle_deg: float,
    stator_indices: np.ndarray,
    rotor_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Pure function: no side effects, no global writes.

    Returns:
        edge_index: [2, E_dyn], int64
        edge_attr:  [E_dyn, 1], float32  (distance-based weight, interior sign=+1)
    """
```

**Rules:**
1. Pure function — no writes to shared state
2. Compatible with existing `static_edge_index` concatenation
3. Must not add duplicate edges with static topology

---

## Memory Safety Contract

Under long dataloader iteration (> 1000 steps):
- `torch.utils.data.DataLoader` with `num_workers ≥ 1` → pin_memory compatible
- No per-worker Python object accumulation (close open handles in worker init)
- Profiler harness: `tests/test_phase2_memory_profile.py`
  - Load 100 batches sequentially
  - Assert GPU memory increase < 200 MB after 100 steps

---

## Normalization Contract

Use per-feature channel normalization computed on training split only:

```python
@dataclass(frozen=True)
class NormStats:
    mean: np.ndarray   # shape [C]
    std: np.ndarray    # shape [C]
    epsilon: float = 1e-8

def normalize(x: np.ndarray, stats: NormStats) -> np.ndarray:
    return (x - stats.mean) / (stats.std + stats.epsilon)
```

- Compute once on train split, persist to `results/norm_stats.json`
- Apply identically to val/test (no refit)
- Inverse-normalize predictions before loss comparison with physical targets

---

## Train Sequence Contract

`train.py` must support `--sequence-len INT` flag:

```
--sequence-len 1   → static (Phase 1 compatible)
--sequence-len T   → temporal (Phase 2, T > 1)
```

When `--sequence-len > 1`:
- DataLoader yields `TemporalMotorSample` batches
- Model receives `(x_t, edge_index_t)` per step `t ∈ 0..T-1`
- Loss is averaged across T steps

---

## Verification Checklist (Phase 2 exit gate)

- [ ] `TemporalMotorSample` dataclass with frozen=True and shape invariants
- [ ] `build_sliding_band_edges()` passes pure-function contract test
- [ ] Memory leak test: < 200 MB increase over 100 batches
- [ ] `--sequence-len` flag plumbed through train.py
- [ ] Normalization stats computed, saved, and loaded reproducibly
- [ ] Interpolated rotor angles do not produce boundary discontinuities (visual check)
- [ ] 3-step rollout MSE trend logged and not diverging

---

## Non-Goals for Phase 2

- Autoregressive rollout stability → Phase 3
- Teacher forcing schedule → Phase 3
- Multi-step training loss (> 5 rollout steps) → Phase 3
- eMach dependency → never in postproc_interop or phase2_dynamic

---

## Agent Handoff Notes

- Do not break `phase1_static/` interfaces when adding `phase2_dynamic/`
- `--sequence-len 1` must remain backward-compatible with Phase 1 checkpoints
- Keep this document updated when contracts change (especially `TemporalMotorSample` schema)
