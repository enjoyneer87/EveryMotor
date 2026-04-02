"""Custom MeshGraphNet message passing with anti-periodic boundary handling."""

from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import MessagePassing


class AntiPeriodicMessagePassing(MessagePassing):
    """MessagePassing block that multiplies edge messages by edge_attr sign."""

    def __init__(self, in_channels: int, out_channels: int, aggr: str = "add") -> None:
        super().__init__(aggr=aggr)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i: torch.Tensor, x_j: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        feat = torch.cat([x_i, x_j, edge_attr], dim=-1)
        msg = self.mlp(feat)
        signed_attr = edge_attr
        if signed_attr.dim() == 1:
            signed_attr = signed_attr.unsqueeze(-1)
        return msg * signed_attr


def apply_edge_sign(messages: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
    """Utility to broadcast edge_attr sign onto precomputed messages."""
    if edge_attr.dim() == 1:
        edge_attr = edge_attr.unsqueeze(-1)
    if edge_attr.shape[0] != messages.shape[0]:
        raise ValueError("edge_attr length must match messages length")
    return messages * edge_attr

