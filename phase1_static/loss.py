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
    if pred.dim() != 2 or target.dim() != 2 or pred.shape[1] < 3 or target.shape[1] < 3:
        raise ValueError("pred/target must have shape [N, >=3] with channels [Bx, By, A, ...]")

    pred_b = pred[:, 0:2]
    target_b = target[:, 0:2]
    pred_a = pred[:, 2:3]
    target_a = target[:, 2:3]

    a_loss = F.mse_loss(pred_a, target_a)
    b_loss = F.mse_loss(pred_b, target_b)

    ones = torch.ones_like(pred_a)
    grads = torch.autograd.grad(
        outputs=pred_a,
        inputs=coords,
        grad_outputs=ones,
        create_graph=True,
        retain_graph=retain_graph,
        only_inputs=True,
        allow_unused=False,
    )[0]

    dA_dx = grads[:, 0:1]
    dA_dy = grads[:, 1:2]
    pred_bx = dA_dy
    pred_by = -dA_dx
    pred_b = torch.cat([pred_bx, pred_by], dim=1)

    curl_loss = F.mse_loss(pred_b, target_b)
    total_loss = (float(w_a) * a_loss) + (float(w_b) * b_loss) + (float(w_curl) * curl_loss)
    metrics = {
        "total_loss": total_loss.detach(),
        "a_loss": a_loss.detach(),
        "b_loss": b_loss.detach(),
        "curl_loss": curl_loss.detach(),
    }
    return total_loss, metrics

