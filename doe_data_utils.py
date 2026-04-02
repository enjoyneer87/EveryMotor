#!/usr/bin/env python3
"""
Shared data utilities for DOE Motor FEA multi-model training.
Reuses the parsing/graph-building logic from train_doe_meshgraphnet.py,
and adds grid interpolation for FNO/RNN models.

Used by: train_doe_fno.py, train_doe_gino.py, train_doe_rnn.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
from scipy.interpolate import griddata
from torch_geometric.data import Data

# PBC utilities (imported lazily to avoid hard dependency when use_pbc=False)
try:
    from pbc_utils import identify_boundary_nodes, find_pbc_edge_pairs
    _PBC_AVAILABLE = True
except ImportError:
    _PBC_AVAILABLE = False

# ---------------------------------------------------------------------------
# H5 parser  (identical to train_doe_meshgraphnet.py)
# ---------------------------------------------------------------------------

def parse_h5_timeseries(path: Path, max_steps: Optional[int] = None) -> List[dict]:
    """Parse a pyMCAD magnetic timeseries H5 file into record dicts."""
    records = []
    with h5py.File(path, "r") as f:
        if "steps" not in f:
            raise ValueError(f"Invalid H5 (missing 'steps'): {path}")

        steps = np.asarray(f["steps"][:], dtype=np.int32)
        node_id = np.asarray(f["mesh/node_id"][:], dtype=np.int32)
        node_x0 = np.asarray(f["mesh/node_x_mm"][:], dtype=np.float64)
        node_y0 = np.asarray(f["mesh/node_y_mm"][:], dtype=np.float64)

        node_1 = np.asarray(f["mesh/node_1"][:], dtype=np.int32)
        node_2 = np.asarray(f["mesh/node_2"][:], dtype=np.int32)
        node_3 = np.asarray(f["mesh/node_3"][:], dtype=np.int32)
        reg_code = np.asarray(f["mesh/reg_code"][:], dtype=np.int32)

        bx_mat = np.asarray(f["fields/bx"][:], dtype=np.float32)
        by_mat = np.asarray(f["fields/by"][:], dtype=np.float32)
        a_mat = np.asarray(f["fields/a"][:], dtype=np.float32) if "fields/a" in f else np.zeros_like(bx_mat)
        j_mat = np.asarray(f["fields/j"][:], dtype=np.float32) if "fields/j" in f else np.zeros_like(bx_mat)

        meta_time = np.asarray(f["meta/time_s"][:], dtype=np.float64) if "meta/time_s" in f else None
        meta_rot = np.asarray(f["meta/rotate_step"][:], dtype=np.float64) if "meta/rotate_step" in f else None

        x_bsm = np.asarray(f["mesh/node_x_mm_by_step_moving"][:], dtype=np.float64) \
            if "mesh/node_x_mm_by_step_moving" in f else None
        y_bsm = np.asarray(f["mesh/node_y_mm_by_step_moving"][:], dtype=np.float64) \
            if "mesh/node_y_mm_by_step_moving" in f else None
        moving_idx = np.asarray(f["mesh/moving_node_indices"][:], dtype=np.int32) \
            if "mesh/moving_node_indices" in f else None

        x_bs = np.asarray(f["mesh/node_x_mm_by_step"][:], dtype=np.float64) \
            if "mesh/node_x_mm_by_step" in f else None
        y_bs = np.asarray(f["mesh/node_y_mm_by_step"][:], dtype=np.float64) \
            if "mesh/node_y_mm_by_step" in f else None

        n_nodes = node_id.size
        sorted_ids = np.sort(node_id)
        max_nid = int(node_id.max()) + 1
        lut = np.full(max_nid, -1, dtype=np.int64)
        for i, nid in enumerate(sorted_ids):
            lut[nid] = i

        m = int(min(len(node_1), len(node_2), len(node_3), len(reg_code)))
        i1 = lut[np.clip(node_1[:m], 0, max_nid - 1)]
        i2 = lut[np.clip(node_2[:m], 0, max_nid - 1)]
        i3 = lut[np.clip(node_3[:m], 0, max_nid - 1)]
        valid_elem = (i1 >= 0) & (i2 >= 0) & (i3 >= 0)
        i1v, i2v, i3v = i1[valid_elem], i2[valid_elem], i3[valid_elem]
        reg_v = reg_code[:m][valid_elem].astype(np.float32)
        nv = int(valid_elem.sum())

        e_src = np.concatenate([i1v, i2v, i3v, i2v, i3v, i1v])
        e_dst = np.concatenate([i2v, i3v, i1v, i1v, i2v, i3v])
        edge_pairs = np.unique(np.stack([e_src, e_dst], axis=1), axis=0)

        all_idx_3 = np.concatenate([i1v, i2v, i3v])
        n = len(sorted_ids)

        all_reg_3 = np.tile(reg_v.astype(np.int32), 3)
        node_reg = np.zeros(n, np.float32)
        if nv > 0 and len(all_reg_3) > 0:
            max_reg = int(all_reg_3.max()) + 1
            for nidx in np.unique(all_idx_3):
                mask = all_idx_3 == nidx
                bc = np.bincount(all_reg_3[mask], minlength=max_reg)
                node_reg[nidx] = float(np.argmax(bc))

        if moving_idx is not None and moving_idx.size > 0:
            mov_mask = (moving_idx >= 0) & (moving_idx < n_nodes)
        else:
            mov_mask = None

        sort_order = np.argsort(node_id)

        n_steps = min(int(steps.shape[0]), max_steps) if max_steps else int(steps.shape[0])
        for si in range(n_steps):
            x = node_x0.copy()
            y = node_y0.copy()

            if x_bs is not None and y_bs is not None:
                xs = x_bs[si]; ys = y_bs[si]
                ok = np.isfinite(xs) & np.isfinite(ys)
                x[ok] = xs[ok]; y[ok] = ys[ok]

            if x_bsm is not None and y_bsm is not None and mov_mask is not None:
                xs_mov = x_bsm[si]
                ys_mov = y_bsm[si]
                n_mov = min(len(xs_mov), mov_mask.sum())
                valid_mi = np.where(mov_mask)[0][:n_mov]
                node_indices = moving_idx[valid_mi]
                xv = xs_mov[valid_mi]
                yv = ys_mov[valid_mi]
                ok_fin = np.isfinite(xv) & np.isfinite(yv)
                x[node_indices[ok_fin]] = xv[ok_fin]
                y[node_indices[ok_fin]] = yv[ok_fin]

            pos_x = x[sort_order]
            pos_y = y[sort_order]

            _ts = float(meta_time[si]) if meta_time is not None else 0.0
            _rs = float(meta_rot[si]) if meta_rot is not None else 0.0
            records.append({
                "step_key": int(steps[si]),
                "time_s": _ts if np.isfinite(_ts) else 0.0,
                "rotate_step": _rs if np.isfinite(_rs) else 0.0,
                "pos_x": pos_x.astype(np.float32),
                "pos_y": pos_y.astype(np.float32),
                "bx": bx_mat[si], "by": by_mat[si],
                "a": a_mat[si], "j": j_mat[si],
                "_i1v": i1v, "_i2v": i2v, "_i3v": i3v,
                "_valid_elem": valid_elem,
                "_all_idx_3": all_idx_3,
                "_edge_pairs": edge_pairs,
                "_node_reg": node_reg,
                "_n": n,
            })
    return records


# ---------------------------------------------------------------------------
# Element-to-node scatter  (shared utility)
# ---------------------------------------------------------------------------

def scatter_elem_to_node(rec: dict):
    """Scatter element FEA fields to node-averaged values.
    Returns (node_bx, node_by, node_a, node_j) arrays of shape (n,)."""
    n = rec["_n"]
    all_idx = rec["_all_idx_3"]
    valid_elem = rec["_valid_elem"]
    m = len(valid_elem)

    bx_v = rec["bx"][:m][valid_elem].astype(np.float32)
    by_v = rec["by"][:m][valid_elem].astype(np.float32)
    aa_v = rec["a"][:m][valid_elem].astype(np.float32)
    jj_v = rec["j"][:m][valid_elem].astype(np.float32)

    all_bx = np.tile(bx_v, 3)
    all_by = np.tile(by_v, 3)
    all_a = np.tile(aa_v, 3)
    all_j = np.tile(jj_v, 3)

    sum_bx = np.zeros(n, np.float32); np.add.at(sum_bx, all_idx, all_bx)
    sum_by = np.zeros(n, np.float32); np.add.at(sum_by, all_idx, all_by)
    sum_a = np.zeros(n, np.float32); np.add.at(sum_a, all_idx, all_a)
    sum_j = np.zeros(n, np.float32); np.add.at(sum_j, all_idx, all_j)
    cnt = np.zeros(n, np.float32); np.add.at(cnt, all_idx, 1.0)
    cnt = np.clip(cnt, 1.0, None)

    return sum_bx / cnt, sum_by / cnt, sum_a / cnt, sum_j / cnt


# ---------------------------------------------------------------------------
# Grid interpolation for FNO / RNN  (mesh -> regular grid)
# ---------------------------------------------------------------------------

def mesh_to_grid(rec: dict, condition: Dict[str, float],
                 grid_res: int = 64) -> Optional[dict]:
    """Interpolate FEM mesh data onto a regular 2D grid.

    Returns dict with:
        'input':  (C_in, H, W)  float32   input channels
        'target': (2, H, W)     float32   [Bx, By]
    or None on failure.
    """
    n = rec["_n"]
    if n == 0:
        return None

    node_bx, node_by, node_a, node_j = scatter_elem_to_node(rec)
    node_reg = rec["_node_reg"]

    pos_x = rec["pos_x"]
    pos_y = rec["pos_y"]
    points = np.column_stack([pos_x, pos_y])

    # Build regular grid covering the mesh bounding box
    xmin, xmax = pos_x.min(), pos_x.max()
    ymin, ymax = pos_y.min(), pos_y.max()
    gx = np.linspace(xmin, xmax, grid_res)
    gy = np.linspace(ymin, ymax, grid_res)
    grid_x, grid_y = np.meshgrid(gx, gy)  # (H, W)

    # Interpolate each field
    def _interp(values):
        return griddata(points, values, (grid_x, grid_y),
                        method='linear', fill_value=0.0).astype(np.float32)

    g_bx = _interp(node_bx)
    g_by = _interp(node_by)
    g_a = _interp(node_a)
    g_j = _interp(node_j)
    g_reg = _interp(node_reg)

    t_s = float(rec.get("time_s", 0.0))
    rot = float(rec.get("rotate_step", 0.0))

    cond_rb = float(condition.get("Ratio_Bore", 0.0))
    cond_rsd = float(condition.get("Ratio_SlotDepth_ParallelSlot", 0.0))
    cond_ipk = float(condition.get("PeakCurrent", 0.0))
    cond_ph = float(condition.get("PhaseAdvance", 0.0))

    # Input channels: [region, time, rotate, RB, RSD, Ipk, Ph] = 7
    input_grid = np.stack([
        g_reg,                                                   # 0: region code
        np.full((grid_res, grid_res), t_s, np.float32),          # 1: time
        np.full((grid_res, grid_res), rot, np.float32),          # 2: rotate_step
        np.full((grid_res, grid_res), cond_rb, np.float32),      # 3: Ratio_Bore
        np.full((grid_res, grid_res), cond_rsd, np.float32),     # 4: Ratio_SlotDepth
        np.full((grid_res, grid_res), cond_ipk, np.float32),     # 5: PeakCurrent
        np.full((grid_res, grid_res), cond_ph, np.float32),      # 6: PhaseAdvance
    ], axis=0)  # (7, H, W)

    target_grid = np.stack([g_bx, g_by, g_a, g_j], axis=0)  # (4, H, W)

    return {
        "input": input_grid,
        "target": target_grid,
        "grid_x": grid_x,
        "grid_y": grid_y,
    }


# ---------------------------------------------------------------------------
# GINO/FNOGNO data format  (mesh coordinates + features)
# ---------------------------------------------------------------------------

def build_gino_sample(rec: dict, condition: Dict[str, float],
                      latent_res: int = 32) -> Optional[dict]:
    """Build data sample for GINO/FNOGNO model.

    Returns dict with:
        'input_geom':     (N, 2)      node coordinates (normalized to [0,1])
        'features':       (N, C_in)   node features (excluding pos)
        'target':         (N, 2)      [Bx, By]
        'latent_queries': (R, R, 2)   regular grid in [0,1]^2
    or None on failure.
    """
    n = rec["_n"]
    if n == 0:
        return None

    node_bx, node_by, node_a, node_j = scatter_elem_to_node(rec)
    node_reg = rec["_node_reg"]

    pos_x = rec["pos_x"]
    pos_y = rec["pos_y"]

    # Normalize coordinates to [0, 1]
    xmin, xmax = pos_x.min(), pos_x.max()
    ymin, ymax = pos_y.min(), pos_y.max()
    x_range = max(xmax - xmin, 1e-6)
    y_range = max(ymax - ymin, 1e-6)
    norm_x = (pos_x - xmin) / x_range
    norm_y = (pos_y - ymin) / y_range

    input_geom = np.column_stack([norm_x, norm_y]).astype(np.float32)  # (N, 2)

    t_s = float(rec.get("time_s", 0.0))
    rot = float(rec.get("rotate_step", 0.0))
    cond_rb = float(condition.get("Ratio_Bore", 0.0))
    cond_rsd = float(condition.get("Ratio_SlotDepth_ParallelSlot", 0.0))
    cond_ipk = float(condition.get("PeakCurrent", 0.0))
    cond_ph = float(condition.get("PhaseAdvance", 0.0))

    # Node features (excluding position): [A, J, reg, time, rot, RB, RSD, Ipk, Ph]
    features = np.column_stack([
        node_reg[:, None],
        np.full((n, 1), t_s, np.float32),
        np.full((n, 1), rot, np.float32),
        np.full((n, 1), cond_rb, np.float32),
        np.full((n, 1), cond_rsd, np.float32),
        np.full((n, 1), cond_ipk, np.float32),
        np.full((n, 1), cond_ph, np.float32),
    ]).astype(np.float32)  # (N, 7)

    target = np.stack([node_bx, node_by, node_a, node_j], axis=1).astype(np.float32)  # (N, 4)

    # Latent grid queries on [0, 1]^2
    lx = np.linspace(0, 1, latent_res, dtype=np.float32)
    ly = np.linspace(0, 1, latent_res, dtype=np.float32)
    lgx, lgy = np.meshgrid(lx, ly)
    latent_queries = np.stack([lgx, lgy], axis=-1)  # (R, R, 2)

    return {
        "input_geom": input_geom,
        "features": features,
        "target": target,
        "latent_queries": latent_queries,
    }


# ---------------------------------------------------------------------------
# Manifest + H5 loader  (shared across all training scripts)
# ---------------------------------------------------------------------------

def load_doe_data(data_dir: str, max_steps_per_case: Optional[int] = None,
                  ) -> Tuple[List[dict], List[Dict[str, float]]]:
    """Load all DOE cases from manifest.

    Returns:
        records:    flat list of per-timestep record dicts
        conditions: parallel list of condition dicts (one per record)
    """
    data_dir = Path(data_dir)
    manifest_path = data_dir / "doe_manifest.json"

    with open(manifest_path) as f:
        manifest = json.load(f)

    print(f"Manifest: {manifest['n_cases']} cases, "
          f"{len(manifest.get('failed', []))} failed")

    all_records: List[dict] = []
    all_conditions: List[Dict[str, float]] = []

    for case in manifest["cases"]:
        h5_paths = case.get("h5_paths") or []
        if not h5_paths:
            continue

        condition = {}
        condition.update(case.get("geometry", {}))
        condition.update(case.get("electrical", {}))

        for h5p in h5_paths:
            h5_basename = h5p.replace("\\", "/").split("/")[-1]
            candidates = [
                Path(h5p),
                data_dir / f"case_{case['index']:04d}" / "postproc" / h5_basename,
                data_dir / h5_basename,
            ]
            h5_file = None
            for c in candidates:
                if c.exists():
                    h5_file = c
                    break

            if h5_file is None:
                continue

            try:
                records = parse_h5_timeseries(h5_file, max_steps=max_steps_per_case)
            except Exception as e:
                print(f"  [WARN] parse error {h5_file}: {e}")
                continue

            for rec in records:
                all_records.append(rec)
                all_conditions.append(condition)

        if len(all_records) > 0:
            idx = case["index"]
            n_new = sum(1 for c in all_conditions if c is condition)
            if n_new > 0:
                print(f"  case {idx:04d}: {n_new} timesteps | "
                      f"RB={condition.get('Ratio_Bore', '?'):.4f} "
                      f"Ipk={condition.get('PeakCurrent', '?'):.1f}")

    print(f"\nTotal records loaded: {len(all_records)}")
    return all_records, all_conditions


# ---------------------------------------------------------------------------
# Graph builder with optional PBC support  (canonical shared version)
# ---------------------------------------------------------------------------

def build_graph(
    rec: dict,
    condition: Dict[str, float],
    use_pbc: bool = False,
    master_angle_deg: float = 0.0,
    slave_angle_deg: float = 45.0,
    pbc_angle_tol_deg: float = 0.5,
    pbc_match_tol: float = 1e-3,
) -> Optional[Data]:
    """Build a torch_geometric.Data graph from one FEA timestep record.

    Node features (x):
        [pos_x, pos_y, region_code, time_s, rotate_step,
         Ratio_Bore, Ratio_SlotDepth, PeakCurrent, PhaseAdvance]
    Target (y):  [Bx, By, A, J]
    Edge attr (use_pbc=False):  [dx, dy, dist]            -- 3 dims (backward-compatible)
    Edge attr (use_pbc=True):   [dx, dy, dist, pbc_flag]  -- 4 dims
        pbc_flag = +1.0 for interior edges, -1.0 for anti-periodic PBC edges.

    Args:
        rec:                Record dict from parse_h5_timeseries.
        condition:          Condition dict with geometry/electrical parameters.
        use_pbc:            Add anti-periodic PBC edges for 1/8 motor model.
                            Default False keeps backward compatibility.
        master_angle_deg:   Angle of master symmetry boundary (degrees).
        slave_angle_deg:    Angle of slave symmetry boundary (degrees).
        pbc_angle_tol_deg:  Angular tolerance for boundary node identification.
        pbc_match_tol:      KDTree distance tolerance for master-slave matching.
    """
    n = rec["_n"]
    if n == 0:
        return None

    pos = np.column_stack([rec["pos_x"], rec["pos_y"]])  # (n, 2)
    edge_pairs = rec["_edge_pairs"]
    if len(edge_pairs) == 0:
        return None

    all_idx = rec["_all_idx_3"]
    node_reg = rec["_node_reg"]
    valid_elem = rec["_valid_elem"]

    bx_raw, by_raw = rec["bx"], rec["by"]
    aa_raw, jj_raw = rec["a"], rec["j"]

    # Scatter element fields → node averages (vectorized)
    m = len(valid_elem)
    bx_v = bx_raw[:m][valid_elem].astype(np.float32)
    by_v = by_raw[:m][valid_elem].astype(np.float32)
    aa_v = aa_raw[:m][valid_elem].astype(np.float32)
    jj_v = jj_raw[:m][valid_elem].astype(np.float32)

    all_bx = np.tile(bx_v, 3)
    all_by = np.tile(by_v, 3)
    all_a  = np.tile(aa_v, 3)
    all_j  = np.tile(jj_v, 3)

    sum_bx = np.zeros(n, np.float32); np.add.at(sum_bx, all_idx, all_bx)
    sum_by = np.zeros(n, np.float32); np.add.at(sum_by, all_idx, all_by)
    sum_a  = np.zeros(n, np.float32); np.add.at(sum_a,  all_idx, all_a)
    sum_j  = np.zeros(n, np.float32); np.add.at(sum_j,  all_idx, all_j)
    cnt    = np.zeros(n, np.float32); np.add.at(cnt,    all_idx, 1.0)
    cnt = np.clip(cnt, 1.0, None)

    node_bx = sum_bx / cnt; node_by = sum_by / cnt
    node_a  = sum_a  / cnt; node_j  = sum_j  / cnt

    t_s = float(rec.get("time_s", 0.0))
    rot = float(rec.get("rotate_step", 0.0))
    cond_rb  = float(condition.get("Ratio_Bore", 0.0))
    cond_rsd = float(condition.get("Ratio_SlotDepth_ParallelSlot", 0.0))
    cond_ipk = float(condition.get("PeakCurrent", 0.0))
    cond_ph  = float(condition.get("PhaseAdvance", 0.0))

    x_feat = np.column_stack([
        pos,                                       # 0,1: x,y
        node_reg[:, None],                         # 2: region code
        np.full((n, 1), t_s,       np.float32),    # 3: time [s]
        np.full((n, 1), rot,       np.float32),    # 4: rotate step
        np.full((n, 1), cond_rb,   np.float32),    # 5: Ratio_Bore
        np.full((n, 1), cond_rsd,  np.float32),    # 6: Ratio_SlotDepth
        np.full((n, 1), cond_ipk,  np.float32),    # 7: PeakCurrent
        np.full((n, 1), cond_ph,   np.float32),    # 8: PhaseAdvance
    ]).astype(np.float32)

    yt = np.stack([node_bx, node_by, node_a, node_j], axis=1).astype(np.float32)

    # Interior edges
    edge_index_np = edge_pairs.T.astype(np.int64)          # (2, E)
    dxy  = pos[edge_index_np[1]] - pos[edge_index_np[0]]
    dist = np.linalg.norm(dxy, axis=1, keepdims=True)

    if not use_pbc:
        edge_attr_np = np.concatenate([dxy, dist], axis=1).astype(np.float32)
        return Data(
            x=torch.from_numpy(x_feat),
            y=torch.from_numpy(yt),
            pos=torch.from_numpy(pos.astype(np.float32)),
            edge_index=torch.from_numpy(edge_index_np),
            edge_attr=torch.from_numpy(edge_attr_np),
        )

    # ---- PBC mode: 4-dim edge_attr [dx, dy, dist, pbc_flag] ----
    if not _PBC_AVAILABLE:
        raise ImportError(
            "pbc_utils.py not found. Cannot use use_pbc=True without pbc_utils."
        )

    ones = np.ones((edge_index_np.shape[1], 1), dtype=np.float32)
    interior_attr = np.concatenate([dxy, dist, ones], axis=1)  # (E_int, 4)

    master_idx, slave_idx = identify_boundary_nodes(
        rec["pos_x"], rec["pos_y"],
        master_angle_deg=master_angle_deg,
        slave_angle_deg=slave_angle_deg,
        angle_tol_deg=pbc_angle_tol_deg,
    )

    pbc_edge_index, pbc_flag_attr = find_pbc_edge_pairs(
        rec["pos_x"], rec["pos_y"],
        master_idx, slave_idx,
        rotation_deg=-(slave_angle_deg - master_angle_deg),
        tol=pbc_match_tol,
    )

    if pbc_edge_index.shape[1] > 0:
        pbc_np = pbc_edge_index.numpy()
        dxy_pbc  = pos[pbc_np[1]] - pos[pbc_np[0]]
        dist_pbc = np.linalg.norm(dxy_pbc, axis=1, keepdims=True)
        pbc_attr_geom = np.concatenate(
            [dxy_pbc, dist_pbc, pbc_flag_attr.numpy()], axis=1
        ).astype(np.float32)

        combined_edge_index = torch.cat(
            [torch.from_numpy(edge_index_np), pbc_edge_index], dim=1
        )
        combined_edge_attr = torch.from_numpy(
            np.concatenate([interior_attr, pbc_attr_geom], axis=0)
        )
    else:
        combined_edge_index = torch.from_numpy(edge_index_np)
        combined_edge_attr  = torch.from_numpy(interior_attr)

    return Data(
        x=torch.from_numpy(x_feat),
        y=torch.from_numpy(yt),
        pos=torch.from_numpy(pos.astype(np.float32)),
        edge_index=combined_edge_index,
        edge_attr=combined_edge_attr,
        master_idx=torch.from_numpy(master_idx.astype(np.int64)),
        slave_idx=torch.from_numpy(slave_idx.astype(np.int64)),
    )
