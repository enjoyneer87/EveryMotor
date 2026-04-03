"""Physics operator abstractions for phase training losses.

The current implementation is 2D-first, with explicit extension seams for 3D.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass(frozen=True)
class ChannelContract:
    """Canonical channel mapping used by the physics operator."""

    b_slice: Tuple[int, int]
    a_index: int
    j_index: int


class PhysicsOperator(ABC):
    """Abstract base for physics-informed differential operators."""

    name: str
    spatial_dim: int
    channel_contract: ChannelContract

    @abstractmethod
    def predict_b_from_pred(self, pred: torch.Tensor, coords: torch.Tensor, *, retain_graph: bool) -> torch.Tensor:
        """Compute B-like components from the model prediction and coordinates."""


class Magnetostatics2DOperator(PhysicsOperator):
    """2D magnetostatics operator: Bx=dA/dy, By=-dA/dx."""

    name = "magnetostatics_2d"
    spatial_dim = 2
    channel_contract = ChannelContract(b_slice=(0, 2), a_index=2, j_index=3)

    def predict_b_from_pred(self, pred: torch.Tensor, coords: torch.Tensor, *, retain_graph: bool) -> torch.Tensor:
        if coords.dim() != 2 or coords.shape[1] != 2:
            raise ValueError(f"{self.name} requires coords with shape [N,2], got {tuple(coords.shape)}")

        pred_a = pred[:, self.channel_contract.a_index : self.channel_contract.a_index + 1]
        grads = torch.autograd.grad(
            outputs=pred_a,
            inputs=coords,
            grad_outputs=torch.ones_like(pred_a),
            create_graph=True,
            retain_graph=retain_graph,
            only_inputs=True,
            allow_unused=False,
        )[0]

        dA_dx = grads[:, 0:1]
        dA_dy = grads[:, 1:2]
        return torch.cat([dA_dy, -dA_dx], dim=1)


def build_physics_operator(spatial_dim: int) -> PhysicsOperator:
    """Factory for physics operators by spatial dimension.

    2D is supported now; 3D is intentionally left as a future extension seam.
    """
    if int(spatial_dim) == 2:
        return Magnetostatics2DOperator()
    if int(spatial_dim) == 3:
        raise NotImplementedError(
            "3D physics operator is not implemented yet. "
            "The seam is ready via PhysicsOperator/build_physics_operator."
        )
    raise ValueError(f"Unsupported spatial_dim: {spatial_dim}")
