"""Loss functions for Phase 1 static 1/8 motor training.

Hybrid strategy:
- Supervise A directly
- Supervise Bx/By directly
- Enforce curl(A) ~= [Bx, By]
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from .contracts import validate_loss_boundary_inputs
from .physics_operators import PhysicsOperator


def lambda_anneal(epoch: int, warmup_epochs: int, base_lambda: float, max_lambda: float) -> float:
    """Linear warmup for a loss weight."""
    if warmup_epochs <= 0:
        return min(base_lambda, max_lambda)
    scale = min(1.0, float(epoch) / float(warmup_epochs))
    return float(min(max_lambda, base_lambda * (1.0 + scale)))


def hybrid_physics_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    coords: torch.Tensor,
    operator: PhysicsOperator,
    *,
    w_a: float = 1.0,
    w_b: float = 1.0,
    w_curl: float = 0.1,
    retain_graph: bool = False,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute weighted A + B + curl consistency loss.

    Expects channel order [Bx, By, A, J].
    J is currently excluded from loss, but retained in the target contract.
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

    pred_b_from_operator = operator.predict_b_from_pred(pred=pred, coords=coords, retain_graph=retain_graph)
    curl_loss = F.mse_loss(pred_b_from_operator, target_b)
    total_loss = (float(w_a) * a_loss) + (float(w_b) * b_loss) + (float(w_curl) * curl_loss)
    metrics = {
        "total_loss": total_loss.detach(),
        "a_loss": a_loss.detach(),
        "b_loss": b_loss.detach(),
        "curl_loss": curl_loss.detach(),
    }
    return total_loss, metrics

