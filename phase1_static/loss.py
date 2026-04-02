"""Physics-informed loss functions for 2D magnetostatics."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F


def lambda_anneal(epoch: int, warmup_epochs: int, base_lambda: float, max_lambda: float) -> float:
    """Linear warmup for physics_loss weight."""
    if warmup_epochs <= 0:
        return min(base_lambda, max_lambda)
    scale = min(1.0, float(epoch) / float(warmup_epochs))
    return float(min(max_lambda, base_lambda * (1.0 + scale)))


def physics_informed_loss(
    pred_a: torch.Tensor,
    target_a: torch.Tensor,
    target_b: torch.Tensor,
    coords: torch.Tensor,
    lambda_weight: float = 0.1,
    *,
    retain_graph: bool = False,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute data + physics loss enforcing curl A = B in 2D."""
    data_loss = F.mse_loss(pred_a, target_a)

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

    physics_loss = F.mse_loss(pred_b, target_b)
    total_loss = data_loss + lambda_weight * physics_loss
    metrics = {
        "total_loss": total_loss.detach(),
        "data_loss": data_loss.detach(),
        "physics_loss": physics_loss.detach(),
    }
    return total_loss, metrics

