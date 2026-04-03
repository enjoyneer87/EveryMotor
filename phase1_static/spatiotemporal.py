"""Spatiotemporal extension seams for future time-stepping phases."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple


@dataclass(frozen=True)
class TemporalContract:
    """Immutable temporal contract for sequence expansion."""

    context_steps: int = 1
    pred_steps: int = 1
    stride: int = 1
    target_mode: str = "absolute"  # "absolute" | "delta"


@dataclass(frozen=True)
class SequenceWindow:
    """Pure index window representation for sequence training."""

    context_indices: Tuple[int, ...]
    pred_indices: Tuple[int, ...]


def build_sequence_windows(
    n_steps: int,
    *,
    context_steps: int,
    pred_steps: int,
    stride: int = 1,
) -> List[SequenceWindow]:
    """Build deterministic sliding windows for time-stepping datasets."""
    if n_steps <= 0:
        return []
    if context_steps <= 0 or pred_steps <= 0 or stride <= 0:
        raise ValueError("context_steps/pred_steps/stride must be positive")

    windows: list[SequenceWindow] = []
    start = 0
    end_limit = n_steps - (context_steps + pred_steps) + 1
    while start < end_limit:
        context = tuple(range(start, start + context_steps))
        pred = tuple(range(start + context_steps, start + context_steps + pred_steps))
        windows.append(SequenceWindow(context_indices=context, pred_indices=pred))
        start += stride
    return windows


def group_indices_by_sequence(samples: Sequence[Dict]) -> Dict[int, List[int]]:
    """Group sample indices by sequence_id without mutating the input."""
    out: Dict[int, List[int]] = {}
    for idx, sample in enumerate(samples):
        seq_id = int(sample.get("sequence_id", -1))
        out.setdefault(seq_id, []).append(idx)
    return out
