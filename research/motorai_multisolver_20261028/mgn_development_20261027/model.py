"""Conditional stationary-slot MeshGraphNet; original triangle edges, no kNN.

Auxiliary output is area-projected MS vector B, not Full current density or loss.
The residual head uses pooled latent features. This is not a temporal rollout.
"""
import torch
from torch import nn


def mlp(nin, nout, hidden, normalize=True):
    layers = [nn.Linear(nin, hidden), nn.SiLU(), nn.Linear(hidden, nout)]
    if normalize:
        layers.append(nn.LayerNorm(nout))
    return nn.Sequential(*layers)


class MeshGraphNet(nn.Module):
    def __init__(self, hidden=48, passes=4):
        super().__init__()
        self.node_encoder = mlp(8, hidden, hidden)
        self.edge_encoder = mlp(3, hidden, hidden)
        self.edge_updates = nn.ModuleList([mlp(hidden*3, hidden, hidden) for _ in range(passes)])
        self.node_updates = nn.ModuleList([mlp(hidden*2, hidden, hidden) for _ in range(passes)])
        self.decoder = mlp(hidden, 2, hidden, normalize=False)

    def forward(self, static, edges, edge_geometry, condition, phase):
        batch = len(condition)
        n = len(static)
        global_x = torch.cat([condition, phase.sin()[:, None], phase.cos()[:, None]], dim=1)
        x = torch.cat([static.expand(batch, -1, -1), global_x[:, None].expand(-1, n, -1)], dim=-1)
        h = self.node_encoder(x).reshape(batch*n, -1)
        e = self.edge_encoder(edge_geometry).expand(batch, -1, -1).reshape(-1, h.shape[-1])
        index = edges[None] + torch.arange(batch, device=h.device)[:, None, None]*n
        src, dst = index[:, 0].flatten(), index[:, 1].flatten()
        for edge_update, node_update in zip(self.edge_updates, self.node_updates):
            e = e + edge_update(torch.cat([h[src], h[dst], e], dim=-1))
            messages = torch.zeros_like(h).index_add_(0, dst, e)
            h = h + node_update(torch.cat([h, messages], dim=-1))
        latent = h.reshape(batch, n, -1)
        return self.decoder(latent), latent


def edge_features(static, edges):
    delta = static[edges[1], :2] - static[edges[0], :2]
    return torch.cat([delta, delta.norm(dim=-1, keepdim=True)], dim=-1)
