"""PyTorch Geometric dataset for the static 1/8 motor model."""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch
from torch_geometric.data import Data, Dataset

from .contracts import encode_fidelity_metadata, normalize_channels_to_bx_by_a_j
from .data_preprocessing import (
    build_pbc_edges,
    combine_edges,
    rotate_points,
    to_tensor,
)
from .pbc_boundary import extract_periodic_boundary_groups_from_mesh


LOG = logging.getLogger(__name__)

PBC_SECTOR_ROTATION_DEG = -45.0
PBC_MATCH_ATOL_MM = 5e-2
PBC_MATCH_RTOL = 1e-6


def _empty_np_edge_index() -> np.ndarray:
    return np.zeros((2, 0), dtype=np.int64)


def _empty_np_edge_attr() -> np.ndarray:
    return np.zeros((0, 1), dtype=np.float32)


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


def _extract_triangles_from_record(rec: Dict[str, Any]) -> np.ndarray:
    i1 = np.asarray(rec.get("_i1v", []), dtype=np.int64)
    i2 = np.asarray(rec.get("_i2v", []), dtype=np.int64)
    i3 = np.asarray(rec.get("_i3v", []), dtype=np.int64)
    if i1.size == 0 or i2.size == 0 or i3.size == 0:
        return np.empty((0, 3), dtype=np.int32)

    n = int(min(i1.size, i2.size, i3.size))
    triangles = np.stack([i1[:n], i2[:n], i3[:n]], axis=1)
    return triangles.astype(np.int32, copy=False)


def _wrap_angle_deg(angle_deg: np.ndarray) -> np.ndarray:
    wrapped = (np.asarray(angle_deg, dtype=np.float64) + 180.0) % 360.0 - 180.0
    return wrapped


def _chain_mean_theta_deg(points_xy: np.ndarray, origin_xy: np.ndarray) -> float:
    rel = np.asarray(points_xy, dtype=np.float64) - np.asarray(origin_xy, dtype=np.float64)
    theta = np.degrees(np.arctan2(rel[:, 1], rel[:, 0]))
    theta_wrapped = _wrap_angle_deg(theta)
    theta_rad = np.deg2rad(theta_wrapped)
    mean_rad = math.atan2(np.sin(theta_rad).mean(), np.cos(theta_rad).mean())
    return float(np.degrees(mean_rad))


def _match_radial_chain_nodes_by_radius(
    master_nodes_xy: np.ndarray,
    slave_nodes_xy: np.ndarray,
    origin_xy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fallback matcher for real DOE meshes when KDTree rotation matching fails.

    Radial boundary chains are paired by relative radius ordering from inner to
    outer radius. This is robust to small center estimation noise and different
    node counts across the two sector boundaries.
    """
    master_nodes = np.asarray(master_nodes_xy, dtype=np.float64)
    slave_nodes = np.asarray(slave_nodes_xy, dtype=np.float64)
    origin = np.asarray(origin_xy, dtype=np.float64)

    if master_nodes.shape[0] == 0 or slave_nodes.shape[0] == 0:
        return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64)

    master_order = np.argsort(np.linalg.norm(master_nodes - origin, axis=1))
    slave_order = np.argsort(np.linalg.norm(slave_nodes - origin, axis=1))
    n_pairs = min(master_order.size, slave_order.size)
    if n_pairs == 0:
        return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64)

    master_pick = np.round(np.linspace(0, master_order.size - 1, n_pairs)).astype(np.int64)
    slave_pick = np.round(np.linspace(0, slave_order.size - 1, n_pairs)).astype(np.int64)
    return master_order[master_pick], slave_order[slave_pick]


def build_sector_pbc_edges_from_mesh(
    pos_xy: np.ndarray,
    triangles: np.ndarray,
    *,
    region_code: Optional[np.ndarray] = None,
    moving_reg_codes: Optional[np.ndarray] = None,
    region_name_by_code: Optional[Dict[int, str]] = None,
    rotation_deg: float = PBC_SECTOR_ROTATION_DEG,
    anti_periodic: bool = True,
    atol: float = PBC_MATCH_ATOL_MM,
    rtol: float = PBC_MATCH_RTOL,
) -> tuple[np.ndarray, np.ndarray, Dict[str, float | int | str]]:
    """Build global PBC edges for a 1/8-sector mesh using boundary-chain extraction.

    Returns empty arrays with an error code in diagnostics when matching is not possible.
    """
    diagnostics: Dict[str, float | int | str] = {
        "rotation_deg": float(rotation_deg),
        "radial_chain_count": 0,
        "group_count": 0,
        "match_ratio": 0.0,
        "error_code": "",
        "max_rotation_residual": 0.0,
    }

    pos = np.asarray(pos_xy, dtype=np.float64)
    tri = np.asarray(triangles, dtype=np.int32)
    if tri.size == 0 or pos.shape[0] == 0:
        diagnostics["error_code"] = "E-PBC-BOUNDARY-EMPTY"
        return _empty_np_edge_index(), _empty_np_edge_attr(), diagnostics

    if region_code is None:
        reg = np.zeros((tri.shape[0],), dtype=np.int32)
    else:
        reg = np.asarray(region_code, dtype=np.int32)
        if reg.shape[0] != tri.shape[0]:
            reg = np.zeros((tri.shape[0],), dtype=np.int32)

    group_specs, group_diag, error_code = (
        extract_periodic_boundary_groups_from_mesh(
        pos,
        tri,
        reg,
        moving_reg_codes=(
            np.asarray(moving_reg_codes, dtype=np.int32)
            if moving_reg_codes is not None else None
        ),
        region_name_by_code=dict(region_name_by_code or {}),
        rotation_deg=rotation_deg,
        )
    )
    diagnostics["radial_chain_count"] = int(
        group_diag.get("external_radial_edge_count", 0)
    )
    diagnostics["group_count"] = int(group_diag.get("group_count", 0))
    group_labels = tuple(group_diag.get("group_labels", tuple()))
    if group_labels:
        diagnostics["group_labels"] = ",".join(
            str(label) for label in group_labels
        )

    if not group_specs:
        diagnostics["error_code"] = error_code or "E-PBC-BOUNDARY-EXTRACT"
        return _empty_np_edge_index(), _empty_np_edge_attr(), diagnostics

    origin = np.zeros((2,), dtype=np.float64)
    matched_pairs: list[np.ndarray] = []
    residual_max = 0.0
    total_slave_nodes = 0
    total_matched = 0
    match_modes: list[str] = []

    for _, master_global_idx, slave_global_idx in group_specs:
        master_global_idx = np.asarray(master_global_idx, dtype=np.int64)
        slave_global_idx = np.asarray(slave_global_idx, dtype=np.int64)
        if master_global_idx.size < 2 or slave_global_idx.size < 2:
            continue

        fallback_used = False
        try:
            _, _, match = build_pbc_edges(
                master_nodes=pos[master_global_idx],
                slave_nodes=pos[slave_global_idx],
                angle_deg=rotation_deg,
                atol=atol,
                rtol=rtol,
                anti_periodic=anti_periodic,
                origin_xy=origin,
            )
            matched_master = np.asarray(match.matched_master, dtype=np.int64)
            matched_slave = np.asarray(match.matched_slave, dtype=np.int64)
        except ValueError:
            matched_master, matched_slave = (
                _match_radial_chain_nodes_by_radius(
                    pos[master_global_idx],
                    pos[slave_global_idx],
                    origin,
                )
            )
            fallback_used = True

        if matched_master.size == 0 or matched_slave.size == 0:
            continue

        global_master = master_global_idx[matched_master]
        global_slave = slave_global_idx[matched_slave]
        matched_pairs.append(
            np.column_stack([global_master, global_slave]).astype(
                np.int64,
                copy=False,
            )
        )
        total_slave_nodes += int(slave_global_idx.size)
        total_matched += int(matched_slave.size)
        match_modes.append(
            "radius_fallback" if fallback_used else "kdtree_rotation"
        )

        rotated_slave = rotate_points(
            pos[global_slave],
            rotation_deg,
            origin_xy=origin,
        )
        residual = np.linalg.norm(pos[global_master] - rotated_slave, axis=1)
        if residual.size > 0:
            residual_max = max(residual_max, float(np.max(residual)))

    if not matched_pairs:
        diagnostics["error_code"] = "E-PBC-MATCH-EMPTY"
        return _empty_np_edge_index(), _empty_np_edge_attr(), diagnostics

    forward = np.concatenate(matched_pairs, axis=0)
    backward = np.column_stack([forward[:, 1], forward[:, 0]])
    edge_pairs = np.concatenate([forward, backward], axis=0)
    edge_index = edge_pairs.T.astype(np.int64, copy=False)

    edge_sign = -1.0 if anti_periodic else 1.0
    edge_attr = np.full((edge_pairs.shape[0], 1), edge_sign, dtype=np.float32)

    diagnostics["match_ratio"] = float(
        total_matched / max(1, total_slave_nodes)
    )
    diagnostics["match_mode"] = (
        match_modes[0]
        if len(set(match_modes)) == 1 else "mixed"
    )
    diagnostics["max_rotation_residual"] = float(residual_max)

    return edge_index, edge_attr, diagnostics


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
        step_code, fidelity_code, coupling_code = encode_fidelity_metadata(
            str(s.get("step_semantics", "explicit_time_step")),
            str(s.get("source_file_type", "Unknown")),
            str(s.get("coupling_policy", "weak_coupled")),
        )
        data.step_semantics = torch.tensor([step_code], dtype=torch.long)
        data.fidelity_type = torch.tensor([fidelity_code], dtype=torch.long)
        data.coupling_policy = torch.tensor([coupling_code], dtype=torch.long)
        return data


def build_samples_from_npz(npz_path: str) -> Sequence[Dict[str, Any]]:
    """Load NPZ/PT samples to canonical sample dicts.

    If precomputed ``pbc_edge_index``/``pbc_edge_attr`` exist in the bundle,
    they are preserved and used directly during training.
    """
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
                "pbc_edge_index": arr["pbc_edge_index"][i] if "pbc_edge_index" in arr else _empty_np_edge_index(),
                "pbc_edge_attr": arr["pbc_edge_attr"][i] if "pbc_edge_attr" in arr else _empty_np_edge_attr(),
                "y": arr["y"][i],
                "spatial_dim": 2,
                "fidelity_level": 1,
                "sample_weight": 1.0,
                "step_index": int(step_idx[i]) if step_idx is not None else i,
                "case_idx": int(case_idx[i]) if case_idx is not None else 0,
                "sequence_id": int(case_idx[i]) if case_idx is not None else 0,
                "time_s": float(time_values[i]) if time_values is not None else 0.0,
                "rotate_step": float(rotate_values[i]) if rotate_values is not None else 0.0,
                "step_semantics": "explicit_time_step",
                "coupling_policy": "transient_coupled",
                "source_file_type": "OnLoadTorque",
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


def build_samples_from_doe_manifest(
    data_dir: str,
    max_steps_per_case: Optional[int] = None,
    case_indices: Optional[Sequence[int]] = None,
    source_file_types: Optional[Sequence[str]] = None,
) -> Sequence[Dict[str, Any]]:
    """Build canonical phase samples from DOE manifest + H5 files.

    Args:
        data_dir: DOE root directory containing doe_manifest.json.
        max_steps_per_case: Optional cap for parsed steps per H5 file.
        case_indices: Optional explicit DOE case indices to include.
        source_file_types: Optional MotorCAD filename classes to include.
    """
    try:
        from doe_data_utils import (
            classify_motorcad_h5_filename,
            infer_coupling_policy,
            infer_step_semantics,
            parse_h5_timeseries,
            scatter_elem_to_node,
        )
    except ImportError as exc:
        raise ImportError("doe_data_utils import failed. Run from repository root.") from exc

    root = Path(data_dir)
    manifest_path = root / "doe_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    case_filter = None if case_indices is None else {int(idx) for idx in case_indices}
    source_filter = None if source_file_types is None else {str(name) for name in source_file_types}

    out: list[Dict[str, Any]] = []
    for case in manifest.get("cases", []):
        case_idx = int(case.get("index", -1))
        if case_filter is not None and case_idx not in case_filter:
            continue
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
            source_type = classify_motorcad_h5_filename(str(h5_ref))
            if source_filter is not None and source_type not in source_filter:
                continue
            step_semantics = infer_step_semantics(source_type)
            coupling_policy = infer_coupling_policy(source_type)
            try:
                records = parse_h5_timeseries(h5_path, max_steps=max_steps_per_case)
            except (ValueError, OSError, KeyError) as exc:
                LOG.warning(
                    "Skipping non-timeseries or invalid H5: case=%04d type=%s file=%s reason=%s",
                    case_idx,
                    source_type,
                    h5_path,
                    exc,
                )
                continue

            for rec in records:
                n_nodes = int(rec["_n"])
                if n_nodes <= 0:
                    continue

                pos = np.column_stack([rec["pos_x"], rec["pos_y"]]).astype(np.float32)
                node_reg = np.asarray(rec["_node_reg"], dtype=np.float32).reshape(-1, 1)
                cond_feat = np.tile(cond_vec[None, :], (n_nodes, 1))
                node_type_onehot = np.concatenate([node_reg, cond_feat], axis=1).astype(np.float32)

                node_bx, node_by, node_a, node_j = scatter_elem_to_node(rec)
                y = np.stack([node_bx, node_by, node_a, node_j], axis=1).astype(np.float32)

                edge_pairs = np.asarray(rec["_edge_pairs"], dtype=np.int64)
                if edge_pairs.ndim != 2 or edge_pairs.shape[1] != 2:
                    continue
                interior_edge_index = edge_pairs.T.copy()

                triangles = _extract_triangles_from_record(rec)
                pbc_edge_index, pbc_edge_attr, pbc_diag = (
                    build_sector_pbc_edges_from_mesh(
                    pos_xy=pos,
                    triangles=triangles,
                    region_code=np.asarray(
                        rec.get("_reg_code", []),
                        dtype=np.int32,
                    ),
                    moving_reg_codes=np.asarray(
                        rec.get("_moving_reg_codes", []),
                        dtype=np.int32,
                    ),
                    region_name_by_code=dict(rec.get("_region_name_by_code", {})),
                    rotation_deg=PBC_SECTOR_ROTATION_DEG,
                    anti_periodic=True,
                    )
                )
                if pbc_diag.get("error_code"):
                    LOG.info(
                        "PBC boundary match skipped: case=%04d step=%s reason=%s",
                        case_idx,
                        rec.get("step_key", -1),
                        pbc_diag["error_code"],
                    )

                out.append(
                    {
                        "pos": pos,
                        "node_type_onehot": node_type_onehot,
                        "interior_edge_index": interior_edge_index,
                        "pbc_edge_index": pbc_edge_index,
                        "pbc_edge_attr": pbc_edge_attr,
                        "y": y,
                        "spatial_dim": 2,
                        "fidelity_level": 2,
                        "sample_weight": 1.0,
                        "case_idx": case_idx,
                        "step_index": int(rec.get("step_key", -1)),
                        "sequence_id": case_idx,
                        "time_s": float(rec.get("time_s", 0.0)),
                        "rotate_step": float(rec.get("rotate_step", 0.0)),
                        "step_semantics": step_semantics,
                        "coupling_policy": coupling_policy,
                        "source_file_type": source_type,
                        "source_file_name": Path(str(h5_ref).replace("\\", "/")).name,
                        "pbc_match_ratio": float(pbc_diag.get("match_ratio", 0.0)),
                        "pbc_rotation_deg": float(pbc_diag.get("rotation_deg", PBC_SECTOR_ROTATION_DEG)),
                        "pbc_error_code": str(pbc_diag.get("error_code", "")),
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
