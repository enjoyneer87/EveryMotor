#!/usr/bin/env python3
"""
Physics-Informed Loss for 2D Magnetostatics (1/8 motor model).

PhysicsInformedLoss implements:
    data_loss    = MSE(pred_A,  true_A)
    physics_loss = MSE(Bx_pred, true_Bx) + MSE(By_pred, true_By)

where Bx_pred, By_pred are derived from curl A via autograd:
    Bx = ∂A/∂y
    By = -∂A/∂x

The caller must ensure:
    - `coords` (node x,y positions) is a leaf tensor with requires_grad=True.
    - `pred_A` is computed from the model whose inputs include `coords`,
      so that the autograd graph connects pred_A → coords.

Lambda (λ) annealing:
    lambda_annealing(epoch, max_epoch, lambda_max)
    returns 0 at epoch 0, ramps linearly to lambda_max by max_epoch.
    Use this to start training on data_loss only, then gradually introduce
    physics_loss for a more stable solution.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Lambda annealing schedule
# ---------------------------------------------------------------------------

def lambda_annealing(epoch: int, max_epoch: int, lambda_max: float = 1.0) -> float:
    """Linear lambda annealing: 0 at epoch 0, lambda_max at max_epoch."""
    if max_epoch <= 0:
        return float(lambda_max)
    return float(lambda_max) * min(1.0, epoch / max_epoch)


# ---------------------------------------------------------------------------
# Physics-Informed Loss
# ---------------------------------------------------------------------------

class PhysicsInformedLoss(nn.Module):
    """Physics-Informed Loss combining data fidelity and ∇×A = B constraint.

    Args:
        lambda_physics:  Default weight for physics_loss when not using annealing.

    Usage:
        criterion = PhysicsInformedLoss(lambda_physics=1.0)

        # In training loop:
        coords = batch.pos.detach().requires_grad_(True)   # [N, 2]
        x = torch.cat([coords, batch.x[:, 2:]], dim=1)    # rebuild with grad coords
        pred = model(x, batch.edge_attr, batch)            # [N, output_dim]
        pred_A = pred[:, 0:1]                              # magnetic vector potential

        total, data_l, phys_l = criterion(
            pred_A     = pred_A,
            true_A     = batch.y[:, 0:1],
            coords     = coords,
            true_Bx    = batch.y[:, 1:2],
            true_By    = batch.y[:, 2:3],
            lambda_val = lambda_annealing(epoch, max_epoch),
        )
    """

    def __init__(self, lambda_physics: float = 1.0) -> None:
        super().__init__()
        self.lambda_physics = lambda_physics

    def forward(
        self,
        pred_A: torch.Tensor,     # [N, 1]  predicted magnetic vector potential
        true_A: torch.Tensor,     # [N, 1]  FEM ground truth A
        coords: torch.Tensor,     # [N, 2]  node positions, requires_grad=True
        true_Bx: torch.Tensor,    # [N, 1]  FEM ground truth Bx
        true_By: torch.Tensor,    # [N, 1]  FEM ground truth By
        lambda_val: float = None,
    ):
        """Compute total, data, and physics losses.

        Returns:
            total_loss:   Scalar tensor (backprop-ready).
            data_loss:    Scalar tensor (detached for logging).
            physics_loss: Scalar tensor (detached for logging).
        """
        lam = float(lambda_val) if lambda_val is not None else self.lambda_physics

        # --- Data loss ---
        data_loss = F.mse_loss(pred_A, true_A)

        # --- Physics loss via autograd ---
        # Ensure the computation graph connects pred_A to coords
        if not coords.requires_grad:
            raise RuntimeError(
                "PhysicsInformedLoss: coords must have requires_grad=True. "
                "Use coords = batch.pos.detach().requires_grad_(True) before forward."
            )

        # dA/d(x,y): shape [N, 2]
        # .sum() reduces to a scalar so autograd computes the Jacobian row-sum,
        # which equals elementwise ∂A_i/∂coord_i for independent node-wise A
        # (valid because node i's A only depends on coord_i through the GNN).
        dA_dxy = torch.autograd.grad(
            outputs=pred_A.sum(),
            inputs=coords,
            create_graph=True,    # needed so physics_loss can be backprop'd
            retain_graph=True,
            allow_unused=False,
        )[0]  # [N, 2]

        # 2D curl:  Bx = ∂A/∂y,  By = -∂A/∂x
        Bx_pred = dA_dxy[:, 1:2]    # [N, 1]
        By_pred = -dA_dxy[:, 0:1]   # [N, 1]

        phys_loss = F.mse_loss(Bx_pred, true_Bx) + F.mse_loss(By_pred, true_By)

        total_loss = data_loss + lam * phys_loss

        return total_loss, data_loss.detach(), phys_loss.detach()
