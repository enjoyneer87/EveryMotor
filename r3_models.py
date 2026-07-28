"""Model factory for the curl-MGN pipeline.

Single source of truth for constructing the field model, shared by the trainer
(``train_doe_curl_mgn.py``) and the eval harness (``eval/predictors.py``) so the
architecture is never re-implemented in two places (the pre-R0 pipeline drifted
exactly that way — a 9-feature builder against an 11-feature docstring).

Every model here honours ONE forward contract, identical to physicsnemo's
MeshGraphNet call in this pipeline::

    model(node_feats: (N, C_in), edge_attr: (E, C_e), graph) -> (N, out_dim)

so the loss / curl / band-spectral / PBC code (all computed from the nodal
output ``raw``) is architecture-independent and needs no change.

Architectures:
  - "mgn"        : physicsnemo MeshGraphNet (the proven baseline; unchanged).
  - "transolver" : R3 — physics-attention graph transformer (global receptive
                   field, linear O(N*slice) attention). Edges unused.
"""
from __future__ import annotations

from typing import Mapping

import torch
import torch.nn as nn


class R3Transolver(nn.Module):
    """Transolver wrapped to the curl-MGN forward contract.

    Transolver (arXiv:2402.02366) is a physics-attention transformer for PDEs on
    general meshes: it projects the N nodes onto ``slice_num`` soft slices and
    attends over those, giving every layer a global receptive field at
    O(N*slice_num) cost — no N^2 blow-up. This wrapper adapts its
    ``forward(fx, embedding, time)`` signature to ``forward(x, edge_attr, graph)``.

    Notes
    -----
    * ``edge_attr`` is ignored — attention is global, not edge-based.
    * Assumes **batch size 1** (one graph per forward). Concatenated PyG batches
      would let attention leak across graphs; this is enforced, and the R3 runs
      use batch 1 on the 3090 anyway.
    * ``x`` arrives already normalized by the pipeline (x_mean/x_std), exactly as
      MeshGraphNet receives it — same input statistics.
    * ``use_te=False``: the Transformer-Engine (fp8) path is disabled so the
      eval determinism contract (bitwise-reproducible scorecards) holds.
    """

    def __init__(
        self,
        in_node: int,
        out_dim: int,
        n_hidden: int = 256,
        n_layers: int = 12,
        n_head: int = 8,
        slice_num: int = 32,
        emb_dim: int = 64,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        from physicsnemo.models.transolver import Transolver

        self.in_node = in_node
        self.out_dim = out_dim
        # Positional embedding from (pos_x, pos_y) — the first two node features.
        self.pos_embed = nn.Linear(2, emb_dim)
        self.core = Transolver(
            functional_dim=in_node,
            out_dim=out_dim,
            embedding_dim=emb_dim,
            n_layers=n_layers,
            n_hidden=n_hidden,
            n_head=n_head,
            slice_num=slice_num,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            unified_pos=False,
            use_te=False,
            time_input=False,
        )

    def forward(self, x: torch.Tensor, edge_attr: torch.Tensor, graph) -> torch.Tensor:
        ng = int(getattr(graph, "num_graphs", 1) or 1)
        if ng != 1:
            raise RuntimeError(
                f"R3Transolver assumes batch size 1 (got num_graphs={ng}); global "
                "attention would leak across concatenated graphs. Use --batch-size 1."
            )
        fx = x.unsqueeze(0)                              # (1, N, C_in)
        emb = self.pos_embed(x[:, :2]).unsqueeze(0)      # (1, N, emb_dim)
        out = self.core(fx, emb)                         # (1, N, out_dim)
        return out.squeeze(0)                            # (N, out_dim)


class R3Hybrid(nn.Module):
    """HybridMeshGraphNet (MGN + long-range "world" edges) at the curl-MGN contract.

    Local message passing over the triangle mesh is kept intact; a separate
    long-range "world" edge set (airgap-band angular skip connections at slot
    pitch, built in ``build_curl_graph``) is added with its own encoder so the
    band can couple across a slot pitch in one hop instead of ~65. This isolates
    "add long-range coupling" from R3's Transolver (which discarded the mesh).

    physicsnemo's HybridMeshGraphNet takes ``forward(node, mesh_edge_feat,
    world_edge_feat, graph)`` and expects ``graph.edge_index`` to be the
    concatenation ``[mesh | world]`` (verified: it splits by the two feature
    tensors' lengths). The CurlData carries mesh edges in ``edge_index`` and the
    world set in ``world_edge_index`` / ``world_edge_attr``; this wrapper
    concatenates them. Batches fine (edges never cross graphs), but the R3 runs
    use batch 1.
    """

    def __init__(self, in_node, in_edge, out_dim, hidden=208, processor_size=15):
        super().__init__()
        from physicsnemo.models.meshgraphnet import HybridMeshGraphNet

        self.core = HybridMeshGraphNet(
            input_dim_nodes=in_node,
            input_dim_edges=in_edge,
            output_dim=out_dim,
            processor_size=processor_size,
            hidden_dim_processor=hidden,
            hidden_dim_node_encoder=hidden,
            hidden_dim_edge_encoder=hidden,
            hidden_dim_node_decoder=hidden,
            aggregation="sum",
        )

    def forward(self, x, edge_attr, graph):
        from torch_geometric.data import Data

        world_ei = getattr(graph, "world_edge_index", None)
        world_attr = getattr(graph, "world_edge_attr", None)
        if world_ei is None or world_ei.numel() == 0:
            world_ei = torch.zeros((2, 0), dtype=torch.long, device=x.device)
            world_attr = torch.zeros((0, edge_attr.shape[1]), dtype=edge_attr.dtype, device=x.device)
        combined = torch.cat([graph.edge_index, world_ei], dim=1)   # [mesh | world]
        g = Data(edge_index=combined, num_nodes=x.shape[0])
        return self.core(x, edge_attr, world_attr, g)              # (N, out_dim)


def build_curl_model(
    arch: str,
    in_node: int,
    in_edge: int,
    out_dim: int,
    hp: Mapping,
    device,
):
    """Construct the field model.

    Parameters
    ----------
    arch : "mgn" | "transolver"
    in_node, in_edge, out_dim : I/O widths (9 / 4 / {1,2}).
    hp : hyperparameter mapping (the trainer's ``vars(args)`` or a checkpoint's
        stored ``args`` dict). Read keys per architecture, with the same
        defaults the pipeline used before this factory existed.
    """
    arch = (arch or "mgn").lower()

    if arch == "mgn":
        from physicsnemo.models.meshgraphnet import MeshGraphNet

        hidden = int(hp.get("hidden_dim", 128))
        return MeshGraphNet(
            input_dim_nodes=in_node,
            input_dim_edges=in_edge,
            output_dim=out_dim,
            processor_size=int(hp.get("processor_size", 15)),
            hidden_dim_processor=hidden,
            hidden_dim_node_encoder=hidden,
            hidden_dim_edge_encoder=hidden,
            hidden_dim_node_decoder=hidden,
            aggregation="sum",
        ).to(device)

    if arch == "transolver":
        return R3Transolver(
            in_node=in_node,
            out_dim=out_dim,
            n_hidden=int(hp.get("hidden_dim", 256)),
            n_layers=int(hp.get("n_layers", 12)),
            n_head=int(hp.get("n_head", 8)),
            slice_num=int(hp.get("slice_num", 32)),
            emb_dim=int(hp.get("emb_dim", 64)),
        ).to(device)

    if arch == "hybrid":
        return R3Hybrid(
            in_node=in_node,
            in_edge=in_edge,
            out_dim=out_dim,
            hidden=int(hp.get("hidden_dim", 208)),
            processor_size=int(hp.get("processor_size", 15)),
        ).to(device)

    raise ValueError(f"unknown model arch {arch!r} (expected 'mgn', 'transolver', or 'hybrid')")
