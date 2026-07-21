#!/usr/bin/env python3
"""
Multi-case MeshGraphNet training for DOE Motor FEA data.

Loads multiple H5 timeseries (each from a different geometry+electrical condition),
injects per-case condition features (Ratio_Bore, Ratio_SlotDepth, PeakCurrent,
PhaseAdvance) into every node, and trains a single MeshGraphNet to predict
(Bx, By) across all conditions.

Node features are the 9 listed in `eval.predictors.MGN_NODE_FEATURES`. A and J
are NOT inputs — they are solved quantities, and feeding them back would make
validation meaningless. `eval.feature_guard` asserts this at graph-build time.

Train/val split is cut at **case** granularity via `eval.case_split`, sharing the
version-controlled split manifest with `eval/benchmark.py`. Splitting the flat
record list at random puts different rotor angles of the same geometry on both
sides, which measures angle interpolation rather than generalization to a new
design; see `.github/plans/methodology_review_20260720.md` F1a.

Usage (inside Docker):
    python /workspace/train_doe_meshgraphnet.py \
        --data-dir /workspace/doe_data \
        --epochs 60 --batch-size 4 --lr 1e-3

Dependencies:  torch, torch_geometric, physicsnemo, h5py, numpy
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from eval.case_split import assign_records, resolve_case_split
from eval.doe_dataset import classify_source_type
from eval.feature_guard import (
    MGN_NODE_FEATURES,
    assert_feature_count,
    assert_input_features_clean,
)

# Only the moving-mesh timeseries solves carry a rotor sweep to learn from.
SOURCE_TYPES_FOR_TRAINING = ("OnLoadTorque",)

# ---------------------------------------------------------------------------
# H5 parser (matches pyMCAD magnetic timeseries H5 schema)
# ---------------------------------------------------------------------------

def parse_h5_timeseries(
    path: Path,
    max_steps: Optional[int] = None,
    step_stride: int = 1,
) -> List[dict]:
    """Parse a pyMCAD magnetic timeseries H5 file into record dicts.

    Returns list of dicts with numpy arrays (no Python-level loops per node).

    `step_stride` subsamples the rotor sweep evenly. Prefer it over `max_steps`
    when trading data volume for run time: `max_steps` truncates, so it keeps
    only the first fraction of the electrical cycle and biases the model toward
    one region of rotor angle.
    """
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
        a_mat  = np.asarray(f["fields/a"][:],  dtype=np.float32) if "fields/a" in f else np.zeros_like(bx_mat)
        j_mat  = np.asarray(f["fields/j"][:],  dtype=np.float32) if "fields/j" in f else np.zeros_like(bx_mat)

        meta_time = np.asarray(f["meta/time_s"][:],      dtype=np.float64) if "meta/time_s"      in f else None
        meta_rot  = np.asarray(f["meta/rotate_step"][:],  dtype=np.float64) if "meta/rotate_step"  in f else None

        # moving-node coords
        x_bsm = np.asarray(f["mesh/node_x_mm_by_step_moving"][:], dtype=np.float64) \
            if "mesh/node_x_mm_by_step_moving" in f else None
        y_bsm = np.asarray(f["mesh/node_y_mm_by_step_moving"][:], dtype=np.float64) \
            if "mesh/node_y_mm_by_step_moving" in f else None
        moving_idx = np.asarray(f["mesh/moving_node_indices"][:], dtype=np.int32) \
            if "mesh/moving_node_indices" in f else None

        # full by_step coords (fallback)
        x_bs = np.asarray(f["mesh/node_x_mm_by_step"][:], dtype=np.float64) \
            if "mesh/node_x_mm_by_step" in f else None
        y_bs = np.asarray(f["mesh/node_y_mm_by_step"][:], dtype=np.float64) \
            if "mesh/node_y_mm_by_step" in f else None

        # --- Precompute fixed topology (once per H5) ---
        n_nodes = node_id.size
        sorted_ids = np.sort(node_id)
        # Build LUT: node_id → contiguous index
        max_nid = int(node_id.max()) + 1
        lut = np.full(max_nid, -1, dtype=np.int64)
        for i, nid in enumerate(sorted_ids):
            lut[nid] = i

        # Element indices in local coords (vectorized)
        m = int(min(len(node_1), len(node_2), len(node_3), len(reg_code)))
        i1 = lut[np.clip(node_1[:m], 0, max_nid - 1)]
        i2 = lut[np.clip(node_2[:m], 0, max_nid - 1)]
        i3 = lut[np.clip(node_3[:m], 0, max_nid - 1)]
        valid_elem = (i1 >= 0) & (i2 >= 0) & (i3 >= 0)
        i1v, i2v, i3v = i1[valid_elem], i2[valid_elem], i3[valid_elem]
        reg_v = reg_code[:m][valid_elem].astype(np.float32)
        nv = int(valid_elem.sum())

        # Edge pairs (undirected, deduplicated) — built ONCE
        e_src = np.concatenate([i1v, i2v, i3v, i2v, i3v, i1v])
        e_dst = np.concatenate([i2v, i3v, i1v, i1v, i2v, i3v])
        edge_pairs = np.unique(np.stack([e_src, e_dst], axis=1), axis=0)

        # Scatter index arrays (shared across timesteps)
        all_idx_3 = np.concatenate([i1v, i2v, i3v])  # 3*nv
        n = len(sorted_ids)

        # Region majority vote (shared topology).
        # Counted with a single flattened bincount over (node, region) pairs.
        # The previous per-node mask over the full incidence array was
        # O(n_nodes * 3*n_elements) and dominated the whole load: ~3 minutes per
        # H5 file, so several hours just to reach the first epoch.
        all_reg_3 = np.tile(reg_v.astype(np.int32), 3)
        node_reg = np.zeros(n, np.float32)
        if nv > 0 and len(all_reg_3) > 0:
            max_reg = int(all_reg_3.max()) + 1
            flat = all_idx_3.astype(np.int64) * max_reg + all_reg_3.astype(np.int64)
            counts = np.bincount(flat, minlength=n * max_reg).reshape(n, max_reg)
            node_reg = counts.argmax(axis=1).astype(np.float32)

        # Moving node precompute
        if moving_idx is not None and moving_idx.size > 0:
            mov_mask = (moving_idx >= 0) & (moving_idx < n_nodes)
        else:
            mov_mask = None

        sort_order = np.argsort(node_id)  # precompute sort index

        n_steps = int(steps.shape[0])
        selected = list(range(0, n_steps, max(1, int(step_stride))))
        if max_steps:
            selected = selected[: int(max_steps)]
        for si in selected:
            x = node_x0.copy()
            y = node_y0.copy()

            # Apply full by_step (vectorized)
            if x_bs is not None and y_bs is not None:
                xs = x_bs[si]; ys = y_bs[si]
                ok = np.isfinite(xs) & np.isfinite(ys)
                x[ok] = xs[ok]; y[ok] = ys[ok]

            # Apply moving-node coords (vectorized, simplified)
            if x_bsm is not None and y_bsm is not None and mov_mask is not None:
                xs_mov = x_bsm[si]
                ys_mov = y_bsm[si]
                n_mov = min(len(xs_mov), mov_mask.sum())
                # mov_mask selects valid entries in moving_idx
                valid_mi = np.where(mov_mask)[0][:n_mov]
                node_indices = moving_idx[valid_mi]
                xv = xs_mov[valid_mi]
                yv = ys_mov[valid_mi]
                ok_fin = np.isfinite(xv) & np.isfinite(yv)
                x[node_indices[ok_fin]] = xv[ok_fin]
                y[node_indices[ok_fin]] = yv[ok_fin]

            # Position array (sorted by node_id)
            pos_x = x[sort_order]
            pos_y = y[sort_order]

            records.append({
                "step_key": int(steps[si]),
                "time_s":   float(meta_time[si]) if meta_time is not None else 0.0,
                "rotate_step": float(meta_rot[si]) if meta_rot is not None else 0.0,
                "pos_x": pos_x.astype(np.float32),
                "pos_y": pos_y.astype(np.float32),
                "bx": bx_mat[si], "by": by_mat[si],
                "a":  a_mat[si],  "j":  j_mat[si],
                # Shared topology (reference, not copied)
                "_i1v": i1v, "_i2v": i2v, "_i3v": i3v,
                "_valid_elem": valid_elem,
                "_all_idx_3": all_idx_3,
                "_edge_pairs": edge_pairs,
                "_node_reg": node_reg,
                "_n": n,
            })
    return records


# ---------------------------------------------------------------------------
# Graph builder (with DOE condition injection)
# ---------------------------------------------------------------------------

def build_graph(rec: dict, condition: Dict[str, float]) -> Optional[Data]:
    """Build a torch_geometric.Data graph from one FEA timestep record.

    Uses precomputed topology from parse_h5_timeseries.
    Node features (x), 9 columns — must stay in sync with
    `eval.predictors.MGN_NODE_FEATURES`:
        [pos_x, pos_y, region_code, time_s, rotate_step,
         Ratio_Bore, Ratio_SlotDepth, PeakCurrent, PhaseAdvance]
    Target (y):  [Bx, By]
    Edge attr:   [dx, dy, dist]

    A and J are deliberately absent from x: both are FEM outputs, so using them
    as inputs would leak the answer. `assert_input_features_clean` enforces it.
    """
    n = rec["_n"]
    if n == 0:
        return None

    pos = np.column_stack([rec["pos_x"], rec["pos_y"]])  # (n, 2)
    edge_pairs = rec["_edge_pairs"]
    if len(edge_pairs) == 0:
        return None

    i1v, i2v, i3v = rec["_i1v"], rec["_i2v"], rec["_i3v"]
    all_idx = rec["_all_idx_3"]
    node_reg = rec["_node_reg"]
    valid_elem = rec["_valid_elem"]

    bx_raw, by_raw = rec["bx"], rec["by"]
    aa_raw, jj_raw = rec["a"], rec["j"]

    # Scatter element fields → node averages (vectorized)
    # Apply same valid_elem mask as topology
    m = len(valid_elem)
    bx_v = bx_raw[:m][valid_elem].astype(np.float32)
    by_v = by_raw[:m][valid_elem].astype(np.float32)
    aa_v = aa_raw[:m][valid_elem].astype(np.float32)
    jj_v = jj_raw[:m][valid_elem].astype(np.float32)

    all_bx = np.tile(bx_v.astype(np.float32), 3)
    all_by = np.tile(by_v.astype(np.float32), 3)
    all_a  = np.tile(aa_v.astype(np.float32), 3)
    all_j  = np.tile(jj_v.astype(np.float32), 3)

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

    x = np.column_stack([
        pos,                                       # 0,1: x,y
        node_reg[:, None],                         # 2: region code
        np.full((n,1), t_s, np.float32),           # 3: time [s]
        np.full((n,1), rot, np.float32),           # 4: rotate step
        np.full((n,1), cond_rb,  np.float32),      # 5: Ratio_Bore
        np.full((n,1), cond_rsd, np.float32),      # 6: Ratio_SlotDepth
        np.full((n,1), cond_ipk, np.float32),      # 7: PeakCurrent
        np.full((n,1), cond_ph,  np.float32),      # 8: PhaseAdvance
    ]).astype(np.float32)

    # Superseded layout: it still carries time_s, which train_doe_curl_mgn.py
    # dropped as a shortcut feature (methodology review section 9).
    assert_input_features_clean(MGN_NODE_FEATURES, context="build_graph node features",
                                allow_shortcuts=True)
    assert_feature_count(MGN_NODE_FEATURES, x.shape[1], context="build_graph node features")

    yt = np.stack([node_bx, node_by], axis=1).astype(np.float32)

    edge_index = edge_pairs.T.astype(np.int64)
    dxy = pos[edge_index[1]] - pos[edge_index[0]]
    dist = np.linalg.norm(dxy, axis=1, keepdims=True)
    edge_attr = np.concatenate([dxy, dist], axis=1).astype(np.float32)

    return Data(
        x=torch.from_numpy(x),
        y=torch.from_numpy(yt),
        pos=torch.from_numpy(pos),
        edge_index=torch.from_numpy(edge_index),
        edge_attr=torch.from_numpy(edge_attr),
    )


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train MeshGraphNet on DOE motor FEA data")
    parser.add_argument("--data-dir", type=str, default="/workspace/doe_data")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--processor-size", type=int, default=15,
                        help="Number of message-passing layers")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--max-steps-per-case", type=int, default=None,
                        help="Cap timesteps per H5 (None=all). Truncates, so it "
                             "keeps only the start of the electrical cycle")
    parser.add_argument("--step-stride", type=int, default=1,
                        help="Keep every Nth timestep. Preferred over "
                             "--max-steps-per-case: samples the whole rotor sweep evenly")
    parser.add_argument("--split", type=str, default="eval/splits/doe40_case_split.json",
                        help="Case-level split manifest, shared with eval/benchmark.py "
                             "(created on first use)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ckpt", type=str, default="/workspace/doe_meshgraphnet_ckpt.pt")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    manifest_path = data_dir / "doe_manifest.json"

    # ---- 1. Load manifest ----
    with open(manifest_path) as f:
        manifest = json.load(f)

    print(f"Manifest: {manifest['n_cases']} cases, {len(manifest.get('failed',[]))} failed")

    # ---- 2. Parse all H5 and build graphs ----
    all_graphs: List[Data] = []
    # Case index of every graph, so the split can be cut at case granularity.
    graph_case_index: List[int] = []
    case_counts = []

    for case in manifest["cases"]:
        h5_paths = case.get("h5_paths") or []
        if not h5_paths:
            continue

        # Build condition dict (geometry + electrical)
        condition = {}
        condition.update(case.get("geometry", {}))
        condition.update(case.get("electrical", {}))

        case_graph_count = 0
        for h5p in h5_paths:
            # StaticLoad / StaticOC / StaticLoadInductance use the static-mesh
            # layout (scalar `step`, 1-D fields) that parse_h5_timeseries cannot
            # read. Skipping them by type keeps the log readable instead of
            # emitting a parse warning per file; eval/doe_dataset.py reads both
            # layouts when the benchmark needs them.
            if classify_source_type(h5p) not in SOURCE_TYPES_FOR_TRAINING:
                continue
            # Extract filename robustly (handles Windows paths on Linux)
            # e.g. "D:\\KDH\\...\\Mag_OnLoadTorque_result_1.h5" → "Mag_OnLoadTorque_result_1.h5"
            h5_basename = h5p.replace("\\", "/").split("/")[-1]

            # Try several path strategies
            candidates = [
                Path(h5p),                                                          # exact (works if same OS)
                data_dir / f"case_{case['index']:04d}" / "postproc" / h5_basename,  # reconstructed
                data_dir / h5_basename,                                             # flat
            ]
            h5_file = None
            for c in candidates:
                if c.exists():
                    h5_file = c
                    break

            if h5_file is None:
                print(f"  [WARN] H5 not found: {h5p} (tried {h5_basename})")
                continue

            try:
                records = parse_h5_timeseries(
                    h5_file,
                    max_steps=args.max_steps_per_case,
                    step_stride=args.step_stride,
                )
            except Exception as e:
                print(f"  [WARN] parse error {h5_file}: {e}")
                continue

            for rec in records:
                g = build_graph(rec, condition)
                if g is not None:
                    all_graphs.append(g)
                    graph_case_index.append(int(case["index"]))
                    case_graph_count += 1

        case_counts.append(case_graph_count)
        if case_graph_count > 0:
            print(f"  case {case['index']:04d}: {case_graph_count} graphs "
                  f"(cond: RB={condition.get('Ratio_Bore','?'):.4f} "
                  f"RSD={condition.get('Ratio_SlotDepth_ParallelSlot','?'):.4f} "
                  f"Ipk={condition.get('PeakCurrent','?'):.1f} "
                  f"Ph={condition.get('PhaseAdvance','?'):.1f})")

    n_total = len(all_graphs)
    print(f"\nTotal graphs: {n_total}")
    if n_total < 2:
        print("ERROR: Need at least 2 graphs for training. Exiting.")
        sys.exit(1)

    # ---- 3. Train / Val split (case-level holdout) ----
    # Cut by geometry, never by record: two rotor angles of the same case are
    # near-duplicates, so a record-level shuffle would leak the val geometry into
    # training and report angle interpolation as if it were generalization.
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    split = resolve_case_split(
        manifest,
        sorted(set(graph_case_index)),
        Path(args.split),
        seed=args.seed,
    )
    assigned = assign_records(graph_case_index, split)
    train_idx, val_idx = assigned["train"], assigned["val"]

    print(f"Split manifest: {args.split}")
    print(f"  train cases {len(split.train)} / val {len(split.val)} / test {len(split.test)} "
          f"(test held back for eval/benchmark.py)")
    if assigned["unassigned"].size:
        print(f"  [WARN] {assigned['unassigned'].size} graphs from cases outside the split "
              f"were dropped")
    if train_idx.size == 0 or val_idx.size == 0:
        print("ERROR: case-level split produced an empty train or val subset. Exiting.")
        sys.exit(1)

    train_graphs = [all_graphs[i] for i in train_idx]
    val_graphs   = [all_graphs[i] for i in val_idx]

    train_cases = {graph_case_index[i] for i in train_idx}
    val_cases   = {graph_case_index[i] for i in val_idx}
    assert not (train_cases & val_cases), "case-level holdout violated"

    # ---- 3b. Sanitize NaN / Inf  ----
    def _sanitize(t: torch.Tensor) -> torch.Tensor:
        return torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)

    for g in all_graphs:
        g.x = _sanitize(g.x)
        g.y = _sanitize(g.y)
        g.edge_attr = _sanitize(g.edge_attr)

    # ---- 4. Global normalization (train stats) ----
    x_cat = torch.cat([g.x for g in train_graphs], dim=0)
    y_cat = torch.cat([g.y for g in train_graphs], dim=0)
    e_cat = torch.cat([g.edge_attr for g in train_graphs], dim=0)

    x_mean = x_cat.mean(dim=0, keepdim=True)
    x_std  = x_cat.std(dim=0, keepdim=True).clamp_min(1e-6)
    y_mean = y_cat.mean(dim=0, keepdim=True)
    y_std  = y_cat.std(dim=0, keepdim=True).clamp_min(1e-6)
    e_mean = e_cat.mean(dim=0, keepdim=True)
    e_std  = e_cat.std(dim=0, keepdim=True).clamp_min(1e-6)

    for g in train_graphs + val_graphs:
        g.x = (g.x - x_mean) / x_std
        g.y = (g.y - y_mean) / y_std
        g.edge_attr = (g.edge_attr - e_mean) / e_std

    train_loader = DataLoader(train_graphs, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_graphs,   batch_size=args.batch_size, shuffle=False)

    print(f"Train: {len(train_graphs)} graphs / {len(train_cases)} cases, "
          f"Val: {len(val_graphs)} graphs / {len(val_cases)} cases")
    print(f"Node features: {train_graphs[0].x.shape[1]}, "
          f"Edge features: {train_graphs[0].edge_attr.shape[1]}, "
          f"Targets: {train_graphs[0].y.shape[1]}")

    # ---- 5. Model ----
    from physicsnemo.models.meshgraphnet import MeshGraphNet

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = MeshGraphNet(
        input_dim_nodes=train_graphs[0].x.shape[1],   # 11
        input_dim_edges=train_graphs[0].edge_attr.shape[1],  # 3
        output_dim=train_graphs[0].y.shape[1],         # 2
        processor_size=args.processor_size,
        hidden_dim_processor=args.hidden_dim,
        hidden_dim_node_encoder=args.hidden_dim,
        hidden_dim_edge_encoder=args.hidden_dim,
        hidden_dim_node_decoder=args.hidden_dim,
        aggregation="sum",
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # ---- 6. Training loop ----
    def run_epoch(loader, training=True):
        model.train() if training else model.eval()
        total_loss, n_batch = 0.0, 0
        for batch in loader:
            batch = batch.to(device)
            with torch.set_grad_enabled(training):
                pred = model(batch.x, batch.edge_attr, batch)
                loss = F.mse_loss(pred, batch.y)
                if torch.isnan(loss) or torch.isinf(loss):
                    continue  # skip NaN/Inf batches
                if training:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
            total_loss += loss.item()
            n_batch += 1
        return total_loss / max(n_batch, 1)

    train_hist, val_hist = [], []
    best_val = float("inf")
    t0 = time.time()

    for ep in range(1, args.epochs + 1):
        tr = run_epoch(train_loader, training=True)
        va = run_epoch(val_loader,   training=False)
        scheduler.step()

        train_hist.append(tr)
        val_hist.append(va)

        if va < best_val:
            best_val = va
            # Save best checkpoint
            torch.save({
                "epoch": ep,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "x_mean": x_mean, "x_std": x_std,
                "y_mean": y_mean, "y_std": y_std,
                "e_mean": e_mean, "e_std": e_std,
                "train_hist": train_hist, "val_hist": val_hist,
                "args": vars(args),
                # Recorded so eval/benchmark.py can verify the checkpoint was
                # trained against the split it is being scored on.
                "split": {
                    "granularity": "case",
                    "manifest": str(args.split),
                    "seed": split.seed,
                    "doe_digest": split.digest,
                    "train_cases": list(split.train),
                    "val_cases": list(split.val),
                    "test_cases": list(split.test),
                },
                "node_features": list(MGN_NODE_FEATURES),
            }, args.ckpt)

        if ep == 1 or ep % 5 == 0 or ep == args.epochs:
            elapsed = time.time() - t0
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"  ep {ep:04d}/{args.epochs} | train {tr:.6f} | val {va:.6f} "
                  f"| best_val {best_val:.6f} | lr {lr_now:.2e} | {elapsed:.0f}s")

    total_time = time.time() - t0
    print(f"\nTraining complete: {args.epochs} epochs in {total_time:.1f}s")
    print(f"Best val MSE: {best_val:.6f}")
    print(f"Checkpoint: {args.ckpt}")


if __name__ == "__main__":
    main()
