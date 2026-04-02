"""PyTorch Geometric dataset for the static 1/8 motor model."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Sequence

import torch
from torch_geometric.data import Data, Dataset

from .data_preprocessing import combine_edges, to_tensor


def _as_feature(tensor: torch.Tensor, dim: int) -> torch.Tensor:
    if tensor.dim() == dim:
        return tensor
    if tensor.dim() == dim - 1:
        return tensor.unsqueeze(-1)
    raise ValueError(f"Expected tensor with {dim} or {dim-1} dims, got {tensor.shape}")


class StaticMotorDataset(Dataset):
    """Static mesh dataset that returns A/B targets and PBC-aware edges."""

    def __init__(
        self,
        samples: Sequence[Dict[str, Any]],
        *,
        dtype: torch.dtype = torch.float32,
        transform=None,
        pre_transform=None,
    ) -> None:
        super().__init__(None, transform, pre_transform)
        self.samples = samples
        self.dtype = dtype

    def len(self) -> int:  # type: ignore[override]
        return len(self.samples)

    def get(self, idx: int) -> Data:  # type: ignore[override]
        s = self.samples[idx]

        pos = to_tensor(s["pos"], dtype=self.dtype)
        pos = _as_feature(pos, 2)
        pos.requires_grad_(True)

        node_type = to_tensor(s.get("node_type_onehot", []), dtype=self.dtype)
        if node_type.numel() == 0:
            node_type = torch.zeros((pos.shape[0], 0), dtype=self.dtype)
        node_type = _as_feature(node_type, 2)

        x = torch.cat([pos, node_type], dim=1)

        interior_edge_index = torch.as_tensor(s["interior_edge_index"], dtype=torch.long)
        pbc_edge_index = torch.as_tensor(s["pbc_edge_index"], dtype=torch.long)
        pbc_edge_attr = torch.as_tensor(s["pbc_edge_attr"], dtype=self.dtype)
        pbc_edge_attr = _as_feature(pbc_edge_attr, 2)

        edge_index, edge_attr = combine_edges(
            interior_edge_index=interior_edge_index,
            pbc_edge_index=pbc_edge_index,
            pbc_edge_attr=pbc_edge_attr,
        )

        if "y" in s:
            y = to_tensor(s["y"], dtype=self.dtype)
        else:
            a = to_tensor(s.get("a"), dtype=self.dtype)
            bx = to_tensor(s.get("b_x") or s.get("bx"), dtype=self.dtype)
            by = to_tensor(s.get("b_y") or s.get("by"), dtype=self.dtype)
            y = torch.cat(
                [
                    _as_feature(a, 2),
                    _as_feature(bx, 2),
                    _as_feature(by, 2),
                ],
                dim=1,
            )

        data = Data(
            x=x,
            pos=pos,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
        )
        return data


def build_samples_from_npz(npz_path: str) -> Sequence[Dict[str, Any]]:
    """Utility loader for simple npz or pt bundles (pos, node_type_onehot, edges, y)."""
    if npz_path.endswith(".pt"):
        bundle = torch.load(npz_path, map_location="cpu")
    else:
        import numpy as _np

        bundle = _np.load(npz_path, allow_pickle=True)

    arr = dict(bundle)
    samples: list[Dict[str, Any]] = []
    n = len(arr["pos"])
    for i in range(n):
        samples.append(
            {
                "pos": arr["pos"][i],
                "node_type_onehot": arr.get("node_type_onehot", [])[i] if "node_type_onehot" in arr else [],
                "interior_edge_index": arr["interior_edge_index"][i],
                "pbc_edge_index": arr["pbc_edge_index"][i],
                "pbc_edge_attr": arr["pbc_edge_attr"][i],
                "y": arr["y"][i],
            }
        )
    return samples
