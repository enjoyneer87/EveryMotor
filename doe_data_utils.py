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


def classify_motorcad_h5_filename(path_like: str) -> str:
    """Classify MotorCAD postprocess file type from filename."""
    name = Path(str(path_like).replace("\\", "/")).name.lower()
    if "onloadtorque" in name:
        return "OnLoadTorque"
    if "losselement" in name or "onloadloss" in name:
        return "LossElement_OnLoadLoss"
    if "staticloadinductance" in name:
        return "StaticLoadInductance"
    if "staticload" in name:
        return "StaticLoad"
    if "staticoc" in name:
        return "StaticOC"
    return "Unknown"


def is_timeseries_h5(path: Path) -> bool:
    """Return True when H5 is supported by training loader."""
    try:
        with h5py.File(path, "r") as f:
            return ("steps" in f) or ("step" in f)
    except OSError:
        return False


def infer_step_semantics(source_type: str) -> str:
    """Infer step semantics by source file type."""
    if source_type == "OnLoadTorque":
        return "explicit_time_step"
    return "implicit_single_step"


def infer_coupling_policy(source_type: str) -> str:
    """Infer coupling policy by source file type."""
    if source_type == "OnLoadTorque":
        return "transient_coupled"
    return "weak_coupled"

# ---------------------------------------------------------------------------
# H5 parser  (identical to train_doe_meshgraphnet.py)
# ---------------------------------------------------------------------------

def parse_h5_timeseries(path: Path, max_steps: Optional[int] = None) -> List[dict]:
    """Parse a pyMCAD magnetic timeseries H5 file into record dicts."""
    records = []
    with h5py.File(path, "r") as f:
        if ("steps" not in f) and ("step" not in f):
            kind = classify_motorcad_h5_filename(str(path))
            raise ValueError(
                f"Invalid H5 (missing 'steps'/'step'): {path} [type={kind}]"
            )

        if "steps" in f:
            steps = np.asarray(f["steps"][:], dtype=np.int32)
        else:
            steps = np.asarray([int(np.asarray(f["step"][()]))], dtype=np.int32)
        node_id = np.asarray(f["mesh/node_id"][:], dtype=np.int32)
        node_x0 = np.asarray(f["mesh/node_x_mm"][:], dtype=np.float64)
        node_y0 = np.asarray(f["mesh/node_y_mm"][:], dtype=np.float64)

        node_1 = np.asarray(f["mesh/node_1"][:], dtype=np.int32)
        node_2 = np.asarray(f["mesh/node_2"][:], dtype=np.int32)
        node_3 = np.asarray(f["mesh/node_3"][:], dtype=np.int32)
        reg_code = np.asarray(f["mesh/reg_code"][:], dtype=np.int32)
        moving_reg_codes = (
            np.asarray(f["mesh/moving_reg_codes"][:], dtype=np.int32)
            if "mesh/moving_reg_codes" in f else np.empty((0,), dtype=np.int32)
        )
        region_name_by_code = {}
        if "regions/reg_code" in f and "regions/name" in f:
            region_codes = np.asarray(f["regions/reg_code"][:], dtype=np.int32)
            region_names = [
                name.decode("utf-8", errors="replace")
                if isinstance(name, bytes) else str(name)
                for name in f["regions/name"][:]
            ]
            region_name_by_code = {
                int(code): str(name)
                for code, name in zip(region_codes.tolist(), region_names)
            }

        bx_raw = np.asarray(f["fields/bx"][:], dtype=np.float32)
        by_raw = np.asarray(f["fields/by"][:], dtype=np.float32)
        a_raw = np.asarray(f["fields/a"][:], dtype=np.float32) if "fields/a" in f else np.zeros_like(bx_raw)
        j_raw = np.asarray(f["fields/j"][:], dtype=np.float32) if "fields/j" in f else np.zeros_like(bx_raw)
        je_raw = np.asarray(f["fields/je"][:], dtype=np.float32) if "fields/je" in f else np.zeros_like(bx_raw)

        bx_mat = bx_raw.reshape(1, -1) if bx_raw.ndim == 1 else bx_raw
        by_mat = by_raw.reshape(1, -1) if by_raw.ndim == 1 else by_raw
        a_mat = a_raw.reshape(1, -1) if a_raw.ndim == 1 else a_raw
        j_mat = j_raw.reshape(1, -1) if j_raw.ndim == 1 else j_raw
        je_mat = je_raw.reshape(1, -1) if je_raw.ndim == 1 else je_raw

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
                "a": a_mat[si], "j": j_mat[si], "je": je_mat[si],
                "_i1v": i1v, "_i2v": i2v, "_i3v": i3v,
                "_reg_code": reg_v.astype(np.int32),
                "_valid_elem": valid_elem,
                "_all_idx_3": all_idx_3,
                "_edge_pairs": edge_pairs,
                "_node_reg": node_reg,
                "_moving_reg_codes": moving_reg_codes,
                "_region_name_by_code": region_name_by_code,
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


def scatter_elem_to_node_with_je(rec: dict):
    """Scatter element FEA fields to node-averaged values including Je.

    Returns (node_bx, node_by, node_a, node_j, node_je) arrays of shape (n,).
    """
    n = rec["_n"]
    all_idx = rec["_all_idx_3"]
    valid_elem = rec["_valid_elem"]
    m = len(valid_elem)

    bx_v = rec["bx"][:m][valid_elem].astype(np.float32)
    by_v = rec["by"][:m][valid_elem].astype(np.float32)
    aa_v = rec["a"][:m][valid_elem].astype(np.float32)
    jj_v = rec["j"][:m][valid_elem].astype(np.float32)
    jje_v = rec.get("je", np.zeros_like(rec["j"]))[:m][valid_elem].astype(np.float32)

    all_bx = np.tile(bx_v, 3)
    all_by = np.tile(by_v, 3)
    all_a = np.tile(aa_v, 3)
    all_j = np.tile(jj_v, 3)
    all_je = np.tile(jje_v, 3)

    sum_bx = np.zeros(n, np.float32); np.add.at(sum_bx, all_idx, all_bx)
    sum_by = np.zeros(n, np.float32); np.add.at(sum_by, all_idx, all_by)
    sum_a = np.zeros(n, np.float32); np.add.at(sum_a, all_idx, all_a)
    sum_j = np.zeros(n, np.float32); np.add.at(sum_j, all_idx, all_j)
    sum_je = np.zeros(n, np.float32); np.add.at(sum_je, all_idx, all_je)
    cnt = np.zeros(n, np.float32); np.add.at(cnt, all_idx, 1.0)
    cnt = np.clip(cnt, 1.0, None)

    return sum_bx / cnt, sum_by / cnt, sum_a / cnt, sum_j / cnt, sum_je / cnt


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
    missing_h5_cases: List[int] = []

    for case in manifest["cases"]:
        h5_paths = case.get("h5_paths") or []
        if not h5_paths:
            txt_paths = case.get("txt_paths") or []
            phases_completed = case.get("phases_completed") or {}
            if txt_paths or phases_completed.get("export_txt"):
                missing_h5_cases.append(int(case.get("index", -1)))
            continue

        condition = {}
        condition.update(case.get("geometry", {}))
        condition.update(case.get("electrical", {}))

        for h5p in h5_paths:
            h5_basename = h5p.replace("\\", "/").split("/")[-1]
            source_type = classify_motorcad_h5_filename(h5p)
            step_semantics = infer_step_semantics(source_type)
            coupling_policy = infer_coupling_policy(source_type)
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

            if not is_timeseries_h5(h5_file):
                print(
                    f"  [INFO] skip non-timeseries H5: {h5_file} "
                    f"(type={source_type})"
                )
                continue

            try:
                records = parse_h5_timeseries(h5_file, max_steps=max_steps_per_case)
            except (ValueError, OSError, KeyError) as e:
                print(f"  [WARN] parse error {h5_file}: {e}")
                continue

            for rec in records:
                rec["source_file_type"] = source_type
                rec["source_file_name"] = h5_basename
                rec["step_semantics"] = step_semantics
                rec["coupling_policy"] = coupling_policy
                all_records.append(rec)
                all_conditions.append(condition)

        if len(all_records) > 0:
            idx = case["index"]
            n_new = sum(1 for c in all_conditions if c is condition)
            if n_new > 0:
                print(f"  case {idx:04d}: {n_new} timesteps | "
                      f"RB={condition.get('Ratio_Bore', '?'):.4f} "
                      f"Ipk={condition.get('PeakCurrent', '?'):.1f}")

    if not all_records and missing_h5_cases:
        case_labels = ", ".join(f"{idx:04d}" for idx in sorted(set(missing_h5_cases)))
        raise ValueError(
            "DOE manifest contains txt-only cases without h5_paths. "
            f"Run the H5 export step first for cases: {case_labels}."
        )

    if missing_h5_cases:
        case_labels = ", ".join(f"{idx:04d}" for idx in sorted(set(missing_h5_cases)))
        print(
            "[WARN] skipping txt-only DOE cases without h5_paths. "
            f"Run the H5 export step first: {case_labels}"
        )

    print(f"\nTotal records loaded: {len(all_records)}")
    return all_records, all_conditions
