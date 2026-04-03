"""PyTorch Geometric dataset for the static 1/8 motor model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch
from torch_geometric.data import Data, Dataset

from .contracts import normalize_channels_to_bx_by_a_j
from .data_preprocessing import combine_edges, to_tensor


def _as_feature(tensor: torch.Tensor, dim: int) -> torch.Tensor:
    if tensor.dim() == dim:
        return tensor
    if tensor.dim() == dim - 1:
        return tensor.unsqueeze(-1)
    raise ValueError(f"Expected tensor with {dim} or {dim-1} dims, got {tensor.shape}")


def _first_present(mapping: Dict[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def _empty_edge_index() -> torch.Tensor:
    return torch.zeros((2, 0), dtype=torch.long)


def _empty_edge_attr(dtype: torch.dtype) -> torch.Tensor:
    return torch.zeros((0, 1), dtype=dtype)


def _onehot_from_labels(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels)
    if labels.size == 0:
        return np.zeros((0, 0), dtype=np.float32)

    uniq = np.unique(labels)
    lut = {value: idx for idx, value in enumerate(uniq.tolist())}
    out = np.zeros((labels.shape[0], len(uniq)), dtype=np.float32)
    for i, value in enumerate(labels.tolist()):
        out[i, lut[value]] = 1.0
    return out


class StaticMotorDataset(Dataset):
    """Static mesh dataset returning canonical target order [Bx, By, A, J]."""

    def __init__(
        self,
        samples: Sequence[Dict[str, Any]],
        *,
        dtype: torch.dtype = torch.float32,
        include_temporal_features: bool = False,
        transform=None,
        pre_transform=None,
    ) -> None:
        super().__init__(None, transform, pre_transform)
        self.samples = samples
        self.dtype = dtype
        self.include_temporal_features = include_temporal_features

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

        x_parts = [pos, node_type]
        if self.include_temporal_features:
            temporal = _temporal_feature_block(s=s, n_nodes=pos.shape[0], dtype=self.dtype)
            x_parts.append(temporal)
        x = torch.cat(x_parts, dim=1)

        interior_edge_index = torch.as_tensor(s["interior_edge_index"], dtype=torch.long)
        pbc_edge_index = torch.as_tensor(s.get("pbc_edge_index", _empty_edge_index()), dtype=torch.long)
        pbc_edge_attr = torch.as_tensor(s.get("pbc_edge_attr", _empty_edge_attr(self.dtype)), dtype=self.dtype)
        pbc_edge_attr = _as_feature(pbc_edge_attr, 2)

        edge_index, edge_attr = combine_edges(
            interior_edge_index=interior_edge_index,
            pbc_edge_index=pbc_edge_index,
            pbc_edge_attr=pbc_edge_attr,
        )

        if "y" in s and s["y"] is not None:
            y_raw = to_tensor(s["y"], dtype=self.dtype)
        else:
            bx_raw = _first_present(s, ("b_x", "bx", "gt_bx_node", "gt_bx"))
            by_raw = _first_present(s, ("b_y", "by", "gt_by_node", "gt_by"))
            a_raw = _first_present(s, ("a", "gt_a_node", "gt_a"))
            j_raw = _first_present(s, ("j", "gt_j_node", "gt_j"))
            if bx_raw is None or by_raw is None:
                raise ValueError("Sample requires either 'y' or explicit Bx/By channels")

            bx = _as_feature(to_tensor(bx_raw, dtype=self.dtype), 2)
            by = _as_feature(to_tensor(by_raw, dtype=self.dtype), 2)
            a = _as_feature(to_tensor(a_raw if a_raw is not None else torch.zeros_like(bx), dtype=self.dtype), 2)
            j = _as_feature(to_tensor(j_raw if j_raw is not None else torch.zeros_like(bx), dtype=self.dtype), 2)
            y_raw = torch.cat([bx, by, a, j], dim=1)

        y, _ = normalize_channels_to_bx_by_a_j(y_raw)

        data = Data(
            x=x,
            pos=pos,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
        )
        data.spatial_dim = torch.tensor([int(s.get("spatial_dim", pos.shape[1]))], dtype=torch.long)
        data.fidelity_level = torch.tensor([int(s.get("fidelity_level", 0))], dtype=torch.long)
        data.sample_weight = torch.tensor([float(s.get("sample_weight", 1.0))], dtype=self.dtype)
        data.step_index = torch.tensor([int(s.get("step_index", -1))], dtype=torch.long)
        data.sequence_id = torch.tensor([int(s.get("sequence_id", -1))], dtype=torch.long)
        return data


def build_samples_from_npz(npz_path: str) -> Sequence[Dict[str, Any]]:
    """Load NPZ/PT samples to canonical sample dicts."""
    if npz_path.endswith(".pt"):
        bundle = torch.load(npz_path, map_location="cpu")
    else:
        bundle = np.load(npz_path, allow_pickle=True)

    arr = dict(bundle)
    samples: list[Dict[str, Any]] = []
    n = len(arr["pos"])
    step_idx = arr["step_idx"] if "step_idx" in arr else None
    case_idx = arr["case_idx"] if "case_idx" in arr else None
    time_values = arr["time_values"] if "time_values" in arr else None
    rotate_values = arr["rotate_values"] if "rotate_values" in arr else None
    for i in range(n):
        samples.append(
            {
                "pos": arr["pos"][i],
                "node_type_onehot": arr.get("node_type_onehot", [])[i] if "node_type_onehot" in arr else [],
                "interior_edge_index": arr["interior_edge_index"][i],
                "pbc_edge_index": arr["pbc_edge_index"][i] if "pbc_edge_index" in arr else np.zeros((2, 0), dtype=np.int64),
                "pbc_edge_attr": arr["pbc_edge_attr"][i] if "pbc_edge_attr" in arr else np.zeros((0, 1), dtype=np.float32),
                "y": arr["y"][i],
                "spatial_dim": 2,
                "fidelity_level": 1,
                "sample_weight": 1.0,
                "step_index": int(step_idx[i]) if step_idx is not None else i,
                "sequence_id": int(case_idx[i]) if case_idx is not None else 0,
                "time_s": float(time_values[i]) if time_values is not None else 0.0,
                "rotate_step": float(rotate_values[i]) if rotate_values is not None else 0.0,
            }
        )
    return samples


def _resolve_h5_candidate(data_dir: Path, case_idx: int, h5_ref: str) -> Optional[Path]:
    basename = h5_ref.replace("\\", "/").split("/")[-1]
    candidates = [
        Path(h5_ref),
        data_dir / f"case_{case_idx:04d}" / "postproc" / basename,
        data_dir / basename,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def build_samples_from_doe_manifest(data_dir: str, max_steps_per_case: Optional[int] = None) -> Sequence[Dict[str, Any]]:
    """Build canonical phase samples from DOE manifest + H5 files."""
    try:
        from doe_data_utils import parse_h5_timeseries, scatter_elem_to_node
    except ImportError as exc:
        raise ImportError("doe_data_utils import failed. Run from repository root.") from exc

    root = Path(data_dir)
    manifest_path = root / "doe_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    out: list[Dict[str, Any]] = []
    for case in manifest.get("cases", []):
        case_idx = int(case.get("index", -1))
        condition = {}
        condition.update(case.get("geometry", {}))
        condition.update(case.get("electrical", {}))
        cond_vec = np.array(
            [
                float(condition.get("Ratio_Bore", 0.0)),
                float(condition.get("Ratio_SlotDepth_ParallelSlot", 0.0)),
                float(condition.get("PeakCurrent", 0.0)),
                float(condition.get("PhaseAdvance", 0.0)),
            ],
            dtype=np.float32,
        )

        for h5_ref in case.get("h5_paths", []) or []:
            h5_path = _resolve_h5_candidate(root, case_idx, str(h5_ref))
            if h5_path is None:
                continue
            records = parse_h5_timeseries(h5_path, max_steps=max_steps_per_case)

            for rec in records:
                n_nodes = int(rec["_n"])
                if n_nodes <= 0:
                    continue

                pos = np.column_stack([rec["pos_x"], rec["pos_y"]]).astype(np.float32)
                node_reg = np.asarray(rec["_node_reg"], dtype=np.int64)
                node_reg_oh = _onehot_from_labels(node_reg)
                cond_feat = np.tile(cond_vec[None, :], (n_nodes, 1))
                node_type_onehot = np.concatenate([node_reg_oh, cond_feat], axis=1).astype(np.float32)

                node_bx, node_by, node_a, node_j = scatter_elem_to_node(rec)
                y = np.stack([node_bx, node_by, node_a, node_j], axis=1).astype(np.float32)

                edge_pairs = np.asarray(rec["_edge_pairs"], dtype=np.int64)
                if edge_pairs.ndim != 2 or edge_pairs.shape[1] != 2:
                    continue
                interior_edge_index = edge_pairs.T.copy()

                out.append(
                    {
                        "pos": pos,
                        "node_type_onehot": node_type_onehot,
                        "interior_edge_index": interior_edge_index,
                        "pbc_edge_index": np.zeros((2, 0), dtype=np.int64),
                        "pbc_edge_attr": np.zeros((0, 1), dtype=np.float32),
                        "y": y,
                        "spatial_dim": 2,
                        "fidelity_level": 2,
                        "sample_weight": 1.0,
                        "step_index": int(rec.get("step_key", -1)),
                        "sequence_id": case_idx,
                        "time_s": float(rec.get("time_s", 0.0)),
                        "rotate_step": float(rec.get("rotate_step", 0.0)),
                    }
                )

    if not out:
        raise RuntimeError(f"No phase samples created from DOE manifest under: {data_dir}")
    return out


def _temporal_feature_block(s: Dict[str, Any], n_nodes: int, dtype: torch.dtype) -> torch.Tensor:
    """Build fixed-width temporal feature block for spatiotemporal expansion.

    Column order: [time_s, rotate_step, dt_s, step_index]
    """
    values = torch.tensor(
        [
            float(s.get("time_s", 0.0)),
            float(s.get("rotate_step", 0.0)),
            float(s.get("dt_s", 0.0)),
            float(s.get("step_index", -1)),
        ],
        dtype=dtype,
    )
    return values.unsqueeze(0).repeat(n_nodes, 1)
