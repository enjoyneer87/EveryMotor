"""Phase 1 contract definitions and pure normalization helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch


TARGET_CHANNEL_ORDER: Tuple[str, str, str, str] = ("Bx", "By", "A", "J")


@dataclass(frozen=True)
class Phase1Contract:
    """Immutable contract declaration for Phase 1 tensor interfaces."""

    target_channel_order: Tuple[str, str, str, str] = TARGET_CHANNEL_ORDER
    interior_edge_sign: float = 1.0
    pbc_anti_periodic_sign: float = -1.0
    spatial_dim: int = 2
    temporal_feature_order: Tuple[str, str, str, str] = ("time_s", "rotate_step", "dt_s", "step_index")


def normalize_channels_to_bx_by_a_j(tensor: torch.Tensor) -> Tuple[torch.Tensor, str]:
    """Normalize tensor channels to canonical order [Bx, By, A, J].

    Supported source layouts:
    - 4+ channels: assumes first four are already [Bx, By, A, J]
    - 3 channels:  [A, Bx, By] -> [Bx, By, A, J=0]
    - 2 channels:  [Bx, By]     -> [Bx, By, A=0, J=0]
    - 1 channel:   [A]          -> [Bx=0, By=0, A, J=0]
    """
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(-1)
    if tensor.dim() != 2:
        raise ValueError(f"Expected tensor with shape [N, C], got {tuple(tensor.shape)}")

    channels = tensor.shape[1]
    if channels >= 4:
        return tensor[:, 0:4], "bx_by_a_j"
    if channels == 3:
        a = tensor[:, 0:1]
        bx = tensor[:, 1:2]
        by = tensor[:, 2:3]
        j = torch.zeros_like(a)
        return torch.cat([bx, by, a, j], dim=1), "a_bx_by"
    if channels == 2:
        bx = tensor[:, 0:1]
        by = tensor[:, 1:2]
        z = torch.zeros_like(bx)
        return torch.cat([bx, by, z, z], dim=1), "bx_by"
    if channels == 1:
        a = tensor[:, 0:1]
        z = torch.zeros_like(a)
        return torch.cat([z, z, a, z], dim=1), "a_only"
    raise ValueError("Channel count must be >= 1")
