#!/usr/bin/env python3
"""
Custom MeshGraphNet with Anti-Periodic Boundary Condition support.

AntiPeriodicMGN implements a full encoder-processor-decoder GNN that handles
anti-periodic PBC edges (1/8 motor model symmetry).

Edge attribute format (when use_pbc=True in build_graph):
    [dx, dy, dist, periodic_flag]
    periodic_flag = +1.0  for normal interior edges
    periodic_flag = -1.0  for anti-periodic PBC edges

During message passing, the source node hidden state (x_j) is multiplied by
the periodic_flag before the edge MLP, so anti-periodic edges propagate a
sign-reversed representation of the neighbour — enforcing A_slave = -A_master.

Interface: compatible with PhysicsNeMo MeshGraphNet
    pred = model(node_features, edge_attr, batch_data)  -> [N, output_dim]
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing


# ---------------------------------------------------------------------------
# MLP helper
# ---------------------------------------------------------------------------

def _mlp(in_dim: int, out_dim: int, hidden_dim: int = None, n_layers: int = 2) -> nn.Sequential:
    """Build a fully-connected MLP with SiLU activations and LayerNorm output."""
    h = hidden_dim or out_dim
    layers: list[nn.Module] = []
    for i in range(n_layers):
        d_in = in_dim if i == 0 else h
        d_out = out_dim if i == n_layers - 1 else h
        layers.append(nn.Linear(d_in, d_out))
        if i < n_layers - 1:
            layers.append(nn.SiLU())
    layers.append(nn.LayerNorm(out_dim))
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# Anti-periodic message passing layer
# ---------------------------------------------------------------------------

class AntiPeriodicMessageLayer(MessagePassing):
    """Single message-passing step with anti-periodic edge support.

    For each edge (i → j):
        sign = edge_sign_flag (scalar ±1.0 per edge)
        msg_ij = EdgeMLP( [h_i, h_j * sign, e_ij] )   -- sign flips h_j on PBC edges
    Node update:
        h'_i = NodeMLP( [h_i, Σ_j msg_ij] ) + h_i     -- residual
    """

    def __init__(self, hidden_dim: int, edge_feat_dim: int) -> None:
        super().__init__(aggr="add")
        self.edge_mlp = _mlp(hidden_dim * 2 + edge_feat_dim, hidden_dim, hidden_dim)
        self.node_mlp = _mlp(hidden_dim * 2, hidden_dim, hidden_dim)

    def forward(
        self,
        h: torch.Tensor,           # [N, H]
        edge_index: torch.Tensor,  # [2, E]
        edge_feats: torch.Tensor,  # [E, edge_feat_dim]  geometric features (no sign)
        sign_flag: torch.Tensor,   # [E, 1]  ±1.0
    ) -> torch.Tensor:
        # propagate returns aggregated messages per node [N, H]
        agg = self.propagate(
            edge_index, x=h, edge_feats=edge_feats, sign_flag=sign_flag
        )
        # Node update with residual connection
        h_new = self.node_mlp(torch.cat([h, agg], dim=-1))
        return h_new + h  # residual

    def message(
        self,
        x_i: torch.Tensor,        # [E, H]  target-node hidden (receiver)
        x_j: torch.Tensor,        # [E, H]  source-node hidden (sender)
        edge_feats: torch.Tensor,  # [E, F]
        sign_flag: torch.Tensor,   # [E, 1]
    ) -> torch.Tensor:
        # Flip the sender's hidden state for anti-periodic edges
        x_j_signed = x_j * sign_flag  # [E, H]
        return self.edge_mlp(torch.cat([x_i, x_j_signed, edge_feats], dim=-1))


# ---------------------------------------------------------------------------
# Full encoder-processor-decoder model
# ---------------------------------------------------------------------------

class AntiPeriodicMGN(nn.Module):
    """Encoder-Processor-Decoder MeshGraphNet with anti-periodic PBC support.

    Args:
        input_dim_nodes:   Number of node feature dimensions.
        input_dim_edges:   Number of edge feature dimensions INCLUDING the sign
                           flag (last column).  Geometric features = input_dim_edges - 1.
        output_dim:        Number of output dimensions per node.
        hidden_dim:        Hidden dimension for all MLPs.
        processor_size:    Number of message-passing layers (processor depth).
    """

    def __init__(
        self,
        input_dim_nodes: int,
        input_dim_edges: int,
        output_dim: int,
        hidden_dim: int = 128,
        processor_size: int = 15,
    ) -> None:
        super().__init__()
        # Sign flag occupies the last column; geometric features are the rest
        self.edge_feat_dim = input_dim_edges - 1

        # Encoder
        self.node_encoder = _mlp(input_dim_nodes, hidden_dim, hidden_dim)
        self.edge_encoder = _mlp(self.edge_feat_dim, hidden_dim, hidden_dim)

        # Processor
        self.processor = nn.ModuleList([
            AntiPeriodicMessageLayer(hidden_dim, hidden_dim)
            for _ in range(processor_size)
        ])

        # Decoder
        self.node_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(
        self,
        x: torch.Tensor,           # [N, input_dim_nodes]
        edge_attr: torch.Tensor,   # [E, input_dim_edges]  last col = sign flag
        graph,                     # torch_geometric.data.Batch / Data  (needs .edge_index)
    ) -> torch.Tensor:             # [N, output_dim]
        edge_index = graph.edge_index                  # [2, E]
        sign_flag = edge_attr[:, -1:]                  # [E, 1]  ±1.0
        edge_feats = edge_attr[:, :-1]                 # [E, edge_feat_dim]

        # Encode
        h = self.node_encoder(x)            # [N, H]
        e = self.edge_encoder(edge_feats)   # [E, H]

        # Process
        for layer in self.processor:
            h = layer(h, edge_index, e, sign_flag)

        # Decode
        return self.node_decoder(h)         # [N, output_dim]
