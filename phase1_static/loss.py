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
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute weighted A + B supervision loss.

    Expects channel order [Bx, By, A, J].
    J is currently excluded from loss, but retained in the target contract.
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

    pred_b = pred[:, b0:b1]
    target_b = target[:, b0:b1]
    pred_a = pred[:, a_index:a_index + 1]
    target_a = target[:, a_index:a_index + 1]

    a_loss = F.mse_loss(pred_a, target_a)
    b_loss = F.mse_loss(pred_b, target_b)

    total_loss = (float(w_a) * a_loss) + (float(w_b) * b_loss)
    metrics = {
        "total_loss": total_loss.detach(),
        "a_loss": a_loss.detach(),
        "b_loss": b_loss.detach(),
    }
    return total_loss, metrics

