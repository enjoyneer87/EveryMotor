"""Loss functions for Phase 1 static 1/8 motor training.

Strategy:
- Supervise A directly (vector potential)
- Supervise Bx/By directly (flux density)
- curl(A) consistency is evaluated in post-processing, not during training.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from .contracts import validate_loss_boundary_inputs
from .physics_operators import PhysicsOperator


def hybrid_physics_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    coords: torch.Tensor,
    operator: PhysicsOperator,
    *,
    w_a: float = 1.0,
    w_b: float = 1.0,
    w_current: float = 1.0,
    current_focus_alpha: float = 2.0,
    current_focus_gamma: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute weighted A + B supervision loss.

    Expects channel order [Bx, By, A, Je].
    Je current-density channel is supervised via current loss.
    curl(A) consistency is NOT computed here — use mesh_edge_curl_b() in
    post-processing for physics consistency checks.
    """
    validate_loss_boundary_inputs(
        pred=pred,
        target=target,
        coords=coords,
        spatial_dim=operator.spatial_dim,
    )

    b0, b1 = operator.channel_contract.b_slice
    a_index = operator.channel_contract.a_index
    c0, c1 = operator.channel_contract.current_density_slice

    pred_b = pred[:, b0:b1]
    target_b = target[:, b0:b1]
    pred_a = pred[:, a_index:a_index + 1]
    target_a = target[:, a_index:a_index + 1]
    pred_current = pred[:, c0:c1]
    target_current = target[:, c0:c1]

    a_loss = F.mse_loss(pred_a, target_a)
    b_loss = F.mse_loss(pred_b, target_b)
    # Je is often spatially sparse (slot-localized). A plain global MSE can
    # under-emphasize those hotspot nodes, so we optionally apply magnitude-
    # aware weighting to focus the current-density loss where |Je| is large.
    if float(current_focus_alpha) > 0.0:
        abs_target = torch.abs(target_current)
        base = abs_target.mean().detach() + 1e-6
        rel = torch.pow(abs_target / base, float(current_focus_gamma))
        weights = 1.0 + float(current_focus_alpha) * rel
        sq_err = torch.pow(pred_current - target_current, 2)
        current_loss = (weights * sq_err).sum() / torch.clamp(weights.sum(), min=1.0)
    else:
        current_loss = F.mse_loss(pred_current, target_current)

    total_loss = (float(w_a) * a_loss) + (float(w_b) * b_loss) + (float(w_current) * current_loss)
    metrics = {
        "total_loss": total_loss.detach(),
        "a_loss": a_loss.detach(),
        "b_loss": b_loss.detach(),
        "current_loss": current_loss.detach(),
    }
    return total_loss, metrics

