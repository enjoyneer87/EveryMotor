#!/usr/bin/env python3
"""
Generate node-wise predictions for all steps of one DOE case.

This script compares multiple models on the same case and stores results on
mesh nodes:
- GT:       Bx, By, A, J
- FNO:      Bx, By, A, J (A/J may be zero if checkpoint output has 2 channels)
- MGN:      Bx, By, A, J (A/J may be zero if checkpoint output has 2 channels)
- SymMGN:   Bx, By, A, J — PBC-aware MeshGraphNet (anti-periodic boundary edges)
- GINO:     Bx, By, A, J (A/J may be zero if checkpoint output has 2 channels)
- RNN:      Bx, By, A, J (A/J may be zero if checkpoint output has 2 channels)

PBC architecture note:
  MGN/FNO/RNN/GINO — trained WITHOUT explicit PBC topology
  SymMGN           — trained WITH anti-periodic PBC edge_attr=-1.0 (phase1_static.train)
                     run via infer_phase1_pbc.py or --symm-mgn-ckpt flag here
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.interpolate import RegularGridInterpolator

sys.path.insert(0, str(Path(__file__).parent))

from doe_data_utils import (
    build_gino_sample,
    mesh_to_grid,
    parse_h5_timeseries,
    scatter_elem_to_node,
)


def sample_grid_to_nodes(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    values: np.ndarray,
    node_x: np.ndarray,
    node_y: np.ndarray,
) -> np.ndarray:
    """Bilinear interpolation from regular grid to node points."""
    gx = grid_x[0, :]
    gy = grid_y[:, 0]
    interp = RegularGridInterpolator(
        (gy, gx),
        values,
        method="linear",
        bounds_error=False,
        fill_value=0.0,
    )
    pts = np.column_stack([node_y, node_x])
    return interp(pts).astype(np.float32)


def sample_4ch_grid_to_nodes(
    gd: dict,
    pred_grid: np.ndarray,
    node_x_step: np.ndarray,
    node_y_step: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample [Bx, By, A, J] from grid to node coordinates."""
    bx = sample_grid_to_nodes(gd["grid_x"], gd["grid_y"], pred_grid[0], node_x_step, node_y_step)
    by = sample_grid_to_nodes(gd["grid_x"], gd["grid_y"], pred_grid[1], node_x_step, node_y_step)
    aa = sample_grid_to_nodes(gd["grid_x"], gd["grid_y"], pred_grid[2], node_x_step, node_y_step)
    jj = sample_grid_to_nodes(gd["grid_x"], gd["grid_y"], pred_grid[3], node_x_step, node_y_step)
    return bx, by, aa, jj


def load_case_records(data_dir: Path, case_idx: int) -> Tuple[dict, list]:
    with open(data_dir / "doe_manifest.json", encoding="utf-8") as f:
        manifest = json.load(f)
    case = manifest["cases"][case_idx]

    cond = {}
    cond.update(case.get("geometry", {}))
    cond.update(case.get("electrical", {}))

    records = []
    for h5p in case.get("h5_paths") or []:
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
            records.extend(parse_h5_timeseries(h5_file))
        except Exception:
            continue

    if not records:
        raise RuntimeError("No records found for case")
    return cond, records


def prepare_case_arrays(records: list, cond: dict, grid_res: int) -> Dict[str, np.ndarray]:
    """Precompute GT node arrays and per-step regular grids."""
    step_count = len(records)
    node_count = records[0]["_n"]

    node_x = np.zeros((step_count, node_count), dtype=np.float32)
    node_y = np.zeros((step_count, node_count), dtype=np.float32)
    gt_bx = np.zeros((step_count, node_count), dtype=np.float32)
    gt_by = np.zeros((step_count, node_count), dtype=np.float32)
    gt_a = np.zeros((step_count, node_count), dtype=np.float32)
    gt_j = np.zeros((step_count, node_count), dtype=np.float32)
    time_values = np.zeros(step_count, dtype=np.float64)
    rotate_values = np.zeros(step_count, dtype=np.float64)
    grids = []

    for si, rec in enumerate(records):
        node_x[si] = rec["pos_x"]
        node_y[si] = rec["pos_y"]
        bx_n, by_n, a_n, j_n = scatter_elem_to_node(rec)
        gt_bx[si] = bx_n
        gt_by[si] = by_n
        gt_a[si] = a_n
        gt_j[si] = j_n
        time_values[si] = rec.get("time_s", 0.0)
        rotate_values[si] = rec.get("rotate_step", 0.0)
        grids.append(mesh_to_grid(rec, cond, grid_res=grid_res))

    return {
        "node_x": node_x,
        "node_y": node_y,
        "gt_bx": gt_bx,
        "gt_by": gt_by,
        "gt_a": gt_a,
        "gt_j": gt_j,
        "time_values": time_values,
        "rotate_values": rotate_values,
        "grids": grids,
    }


def infer_fno(
    grids: list,
    node_x: np.ndarray,
    node_y: np.ndarray,
    ckpt_path: Path,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    from physicsnemo.models.fno import FNO

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    ad = ckpt.get("args", {})
    x_mean = ckpt["x_mean"].to(device)
    x_std = ckpt["x_std"].to(device)
    y_mean = ckpt["y_mean"].to(device)
    y_std = ckpt["y_std"].to(device)

    in_channels = int(x_mean.shape[1])
    out_channels = int(y_mean.shape[1])

    model = FNO(
        in_channels=in_channels,
        out_channels=out_channels,
        dimension=2,
        latent_channels=ad.get("fno_hidden", 64),
        num_fno_layers=ad.get("fno_layers", 4),
        num_fno_modes=[ad.get("fno_modes", 16)] * 2,
        padding=8,
        activation_fn="gelu",
        coord_features=True,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    s, n = node_x.shape
    out = {
        "bx": np.zeros((s, n), dtype=np.float32),
        "by": np.zeros((s, n), dtype=np.float32),
        "a": np.zeros((s, n), dtype=np.float32),
        "j": np.zeros((s, n), dtype=np.float32),
    }

    with torch.no_grad():
        for si, gd in enumerate(grids):
            x = torch.from_numpy(gd["input"]).unsqueeze(0).to(device)
            x = torch.nan_to_num(x)
            y_norm = model((x - x_mean) / x_std)
            y = y_norm * y_std + y_mean

            pred_grid = np.zeros((4, *gd["input"].shape[1:]), dtype=np.float32)
            pred_np = y[0].detach().cpu().numpy()
            pred_grid[: min(4, pred_np.shape[0])] = pred_np[: min(4, pred_np.shape[0])]

            bx, by, aa, jj = sample_4ch_grid_to_nodes(gd, pred_grid, node_x[si], node_y[si])
            out["bx"][si], out["by"][si], out["a"][si], out["j"][si] = bx, by, aa, jj

    return out


def infer_mgn(
    records: list,
    cond: dict,
    ckpt_path: Path,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    from physicsnemo.models.meshgraphnet import MeshGraphNet
    from train_doe_meshgraphnet import build_graph

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    ad = ckpt.get("args", {})
    x_mean = ckpt["x_mean"].to(device)
    x_std = ckpt["x_std"].to(device)
    y_mean = ckpt["y_mean"].to(device)
    y_std = ckpt["y_std"].to(device)
    e_mean = ckpt.get("e_mean", torch.tensor([[0.0, 0.0, 0.0]])).to(device)
    e_std = ckpt.get("e_std", torch.tensor([[1.0, 1.0, 1.0]])).to(device)

    model = MeshGraphNet(
        input_dim_nodes=int(x_mean.shape[1]),
        input_dim_edges=int(e_mean.shape[1]),
        output_dim=int(y_mean.shape[1]),
        processor_size=ad.get("processor_size", 10),
        hidden_dim_processor=ad.get("hidden_dim", 128),
        hidden_dim_node_encoder=ad.get("hidden_dim", 128),
        hidden_dim_edge_encoder=ad.get("hidden_dim", 128),
        hidden_dim_node_decoder=ad.get("hidden_dim", 128),
        aggregation="sum",
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    s = len(records)
    n = records[0]["_n"]
    out = {
        "bx": np.zeros((s, n), dtype=np.float32),
        "by": np.zeros((s, n), dtype=np.float32),
        "a": np.zeros((s, n), dtype=np.float32),
        "j": np.zeros((s, n), dtype=np.float32),
    }

    with torch.no_grad():
        for si, rec in enumerate(records):
            g = build_graph(rec, cond)
            g.x = torch.nan_to_num(g.x)
            g.edge_attr = torch.nan_to_num(g.edge_attr)

            xn = (g.x.to(device) - x_mean) / x_std
            en = (g.edge_attr.to(device) - e_mean) / e_std
            g_dev = g.clone()
            g_dev.x = xn
            g_dev.edge_attr = en
            g_dev.edge_index = g.edge_index.to(device)

            y = model(g_dev.x, g_dev.edge_attr, g_dev) * y_std + y_mean
            pred = y.detach().cpu().numpy()

            out["bx"][si] = pred[:, 0]
            out["by"][si] = pred[:, 1]
            if pred.shape[1] >= 4:
                out["a"][si] = pred[:, 2]
                out["j"][si] = pred[:, 3]

    return out


def infer_gino(
    records: list,
    cond: dict,
    ckpt_path: Path,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    from neuralop.models import GINO

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    ad = ckpt.get("args", {})
    feat_mean = np.asarray(ckpt["feat_mean"], dtype=np.float32)
    feat_std = np.asarray(ckpt["feat_std"], dtype=np.float32)
    tgt_mean = np.asarray(ckpt["tgt_mean"], dtype=np.float32)
    tgt_std = np.asarray(ckpt["tgt_std"], dtype=np.float32)

    in_channels = int(feat_mean.shape[1])
    out_channels = int(tgt_mean.shape[1])

    model = GINO(
        in_channels=in_channels,
        out_channels=out_channels,
        gno_coord_dim=2,
        in_gno_radius=ad.get("gno_radius", 0.1),
        out_gno_radius=ad.get("gno_radius", 0.1),
        fno_in_channels=ad.get("fno_in_channels", in_channels),
        fno_n_modes=(ad.get("fno_modes", 16), ad.get("fno_modes", 16)),
        fno_hidden_channels=ad.get("fno_hidden", 64),
        fno_n_layers=ad.get("fno_layers", 4),
        fno_lifting_channel_ratio=2,
        projection_channel_ratio=4,
        in_gno_transform_type="linear",
        out_gno_transform_type="linear",
        in_gno_pos_embed_type="transformer",
        out_gno_pos_embed_type="transformer",
        gno_embed_channels=16,
        in_gno_channel_mlp_hidden_layers=[80, 80],
        out_gno_channel_mlp_hidden_layers=[256, 128],
        gno_channel_mlp_non_linearity=F.gelu,
        gno_use_open3d=False,
        gno_use_torch_scatter=True,
        fno_use_channel_mlp=True,
        fno_channel_mlp_expansion=0.5,
        fno_non_linearity=F.gelu,
        fno_norm=None,
        fno_skip="linear",
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    s = len(records)
    n = records[0]["_n"]
    out = {
        "bx": np.zeros((s, n), dtype=np.float32),
        "by": np.zeros((s, n), dtype=np.float32),
        "a": np.zeros((s, n), dtype=np.float32),
        "j": np.zeros((s, n), dtype=np.float32),
    }

    with torch.no_grad():
        for si, rec in enumerate(records):
            sample = build_gino_sample(rec, cond, latent_res=ad.get("latent_res", 32))
            feats = ((sample["features"] - feat_mean) / feat_std).astype(np.float32)
            input_geom = torch.from_numpy(sample["input_geom"]).unsqueeze(0).to(device)
            latent_q = torch.from_numpy(sample["latent_queries"]).unsqueeze(0).to(device)
            x_feat = torch.from_numpy(feats).unsqueeze(0).to(device)

            pred_norm = model(
                input_geom=input_geom,
                latent_queries=latent_q,
                output_queries=input_geom,
                x=x_feat,
            )
            pred = pred_norm.squeeze(0).detach().cpu().numpy() * tgt_std + tgt_mean
            out["bx"][si] = pred[:, 0]
            out["by"][si] = pred[:, 1]
            if pred.shape[1] >= 4:
                out["a"][si] = pred[:, 2]
                out["j"][si] = pred[:, 3]

    return out


def infer_rnn(
    grids: list,
    node_x: np.ndarray,
    node_y: np.ndarray,
    ckpt_path: Path,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    from physicsnemo.models.rnn import Seq2SeqRNN

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    ad = ckpt.get("args", {})
    seq_len = int(ad.get("seq_out", 4))

    inp_mean = torch.as_tensor(ckpt["inp_mean"], device=device)
    inp_std = torch.as_tensor(ckpt["inp_std"], device=device)
    tgt_mean = torch.as_tensor(ckpt["tgt_mean"], device=device)
    tgt_std = torch.as_tensor(ckpt["tgt_std"], device=device)

    model = Seq2SeqRNN(
        input_channels=int(inp_mean.shape[0]),
        dimension=2,
        nr_latent_channels=ad.get("hidden_channels", 64),
        nr_tsteps=seq_len,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    s, n = node_x.shape
    out = {
        "bx": np.zeros((s, n), dtype=np.float32),
        "by": np.zeros((s, n), dtype=np.float32),
        "a": np.zeros((s, n), dtype=np.float32),
        "j": np.zeros((s, n), dtype=np.float32),
    }

    frames = []
    for gd in grids:
        frame = np.stack(
            [
                gd["target"][0],
                gd["target"][1],
                gd["input"][0],
                gd["input"][1],
                gd["input"][2],
            ],
            axis=0,
        )
        frames.append(frame)

    with torch.no_grad():
        for si in range(s):
            start = max(0, si - seq_len + 1)
            seq = []
            for ti in range(start, start + seq_len):
                idx = min(ti, s - 1)
                seq.append(frames[idx])
            while len(seq) < seq_len:
                seq.insert(0, seq[0])

            x_seq = np.stack(seq, axis=1)
            x = torch.from_numpy(x_seq).unsqueeze(0).float().to(device)
            x = torch.nan_to_num(x)
            x_norm = (x - inp_mean.unsqueeze(0)) / inp_std.unsqueeze(0)

            pred_full = model(x_norm)
            out_dim = pred_full.shape[1]
            pred = pred_full * tgt_std.unsqueeze(0) + tgt_mean.unsqueeze(0)

            pred_grid = np.zeros((4, *grids[si]["input"].shape[1:]), dtype=np.float32)
            pred_np = pred[0, :, -1].detach().cpu().numpy()
            pred_grid[: min(4, out_dim)] = pred_np[: min(4, out_dim)]

            bx, by, aa, jj = sample_4ch_grid_to_nodes(
                grids[si], pred_grid, node_x[si], node_y[si]
            )
            out["bx"][si], out["by"][si], out["a"][si], out["j"][si] = bx, by, aa, jj

    return out


def infer_symm_mgn(
    records: list,
    cond: dict,
    ckpt_path: Path,
    device: torch.device,
) -> Optional[Dict[str, np.ndarray]]:
    """SymMGN (PBC-aware MeshGraphNet) inference via phase1_static pipeline.

    Returns None if checkpoint is unavailable; caller should treat as skip.
    PBC edges are rebuilt from mesh topology inside StaticMotorDataset.
    """
    try:
        from infer_phase1_pbc import load_symm_mgn, run_inference
        from phase1_static.motor_dataset import build_samples_from_doe_manifest
    except ImportError as exc:
        print(f"[SymMGN] import failed: {exc}")
        return None

    if not ckpt_path.exists():
        print(f"[SymMGN] checkpoint not found: {ckpt_path} — skipped")
        return None

    model, _ = load_symm_mgn(ckpt_path, device)

    # Build phase1_static samples from DOE records
    # Reuse node coordinates and y-targets from pre-parsed records
    samples = []
    for rec in records:
        pos = np.stack([rec["pos_x"], rec["pos_y"]], axis=1).astype(np.float32)
        bx_n, by_n, a_n, j_n = scatter_elem_to_node(rec)
        y = np.stack([bx_n, by_n, a_n, j_n], axis=1).astype(np.float32)
        n = pos.shape[0]
        # Minimal interior edges: use triangle neighbours if available, else self-loops
        tri = rec.get("triangles", None)
        if tri is not None and np.asarray(tri).size > 0:
            tri_np = np.asarray(tri, dtype=np.int32)
            src = np.concatenate([tri_np[:, 0], tri_np[:, 1], tri_np[:, 2]])
            dst = np.concatenate([tri_np[:, 1], tri_np[:, 2], tri_np[:, 0]])
            ie = np.stack([np.concatenate([src, dst]),
                           np.concatenate([dst, src])], axis=0).astype(np.int64)
        else:
            ie = np.zeros((2, 0), dtype=np.int64)
        samples.append({
            "pos": pos,
            "node_type_onehot": np.ones((n, 1), dtype=np.float32),
            "interior_edge_index": ie,
            "y": y,
        })

    result = run_inference(model, samples, device)
    s = len(records)
    n = records[0]["_n"]
    out = {
        "bx": result["pred_bx"].reshape(s, -1)[:, :n],
        "by": result["pred_by"].reshape(s, -1)[:, :n],
        "a":  result["pred_a"].reshape(s, -1)[:, :n],
        "j":  result["pred_j"].reshape(s, -1)[:, :n],
    }
    return out


def save_results_npz(
    out_path: Path,
    case_idx: int,
    cond: dict,
    prepared: Dict[str, np.ndarray],
    fno_pred: Dict[str, np.ndarray],
    mgn_pred: Dict[str, np.ndarray],
    gino_pred: Dict[str, np.ndarray],
    rnn_pred: Dict[str, np.ndarray],
    symm_mgn_pred: Optional[Dict[str, np.ndarray]] = None,
) -> None:
    s, n = prepared["node_x"].shape
    np.savez_compressed(
        out_path,
        case_idx=np.int32(case_idx),
        n_steps=np.int32(s),
        n_nodes=np.int32(n),
        condition=json.dumps(cond),
        time_values=prepared["time_values"],
        rotate_values=prepared["rotate_values"],
        node_x=prepared["node_x"],
        node_y=prepared["node_y"],
        gt_bx_node=prepared["gt_bx"],
        gt_by_node=prepared["gt_by"],
        gt_a_node=prepared["gt_a"],
        gt_j_node=prepared["gt_j"],
        fno_bx_node=fno_pred["bx"],
        fno_by_node=fno_pred["by"],
        fno_a_node=fno_pred["a"],
        fno_j_node=fno_pred["j"],
        mgn_bx_node=mgn_pred["bx"],
        mgn_by_node=mgn_pred["by"],
        mgn_a_node=mgn_pred["a"],
        mgn_j_node=mgn_pred["j"],
        gino_bx_node=gino_pred["bx"],
        gino_by_node=gino_pred["by"],
        gino_a_node=gino_pred["a"],
        gino_j_node=gino_pred["j"],
        gin_a_node=gino_pred["a"],
        gin_j_node=gino_pred["j"],
        rnn_bx_node=rnn_pred["bx"],
        rnn_by_node=rnn_pred["by"],
        rnn_a_node=rnn_pred["a"],
        rnn_j_node=rnn_pred["j"],
        # SymMGN (PBC-aware) — zeros if not run
        symm_mgn_bx_node=symm_mgn_pred["bx"] if symm_mgn_pred else np.zeros_like(prepared["gt_bx"]),
        symm_mgn_by_node=symm_mgn_pred["by"] if symm_mgn_pred else np.zeros_like(prepared["gt_by"]),
        symm_mgn_a_node=symm_mgn_pred["a"]  if symm_mgn_pred else np.zeros_like(prepared["gt_a"]),
        symm_mgn_j_node=symm_mgn_pred["j"]  if symm_mgn_pred else np.zeros_like(prepared["gt_j"]),
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/workspace/host_data/doe_data")
    ap.add_argument("--case-idx", type=int, default=5)
    ap.add_argument("--grid-res", type=int, default=64)
    ap.add_argument("--mgn-ckpt", default="/workspace/host_data/doe_meshgraphnet_ckpt.pt")
    ap.add_argument("--fno-ckpt", default="/workspace/host_data/doe_fno_ckpt.pt")
    ap.add_argument("--gino-ckpt", default="/workspace/host_data/doe_gino_ckpt.pt")
    ap.add_argument("--rnn-ckpt", default="/workspace/host_data/doe_rnn_ckpt.pt")
    ap.add_argument(
        "--symm-mgn-ckpt",
        default=None,
        help="SymMGN (PBC-aware) checkpoint from phase1_static.train --ckpt-out. If omitted, SymMGN is skipped.",
    )
    ap.add_argument("--out", default="/workspace/host_data/field_compare_nodes_allsteps.npz")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    data_dir = Path(args.data_dir)
    cond, records = load_case_records(data_dir, args.case_idx)
    print(f"case={args.case_idx}, steps={len(records)}, nodes={records[0]['_n']}")

    prepared = prepare_case_arrays(records, cond, grid_res=args.grid_res)

    t0 = time.time()
    fno_pred = infer_fno(prepared["grids"], prepared["node_x"], prepared["node_y"], Path(args.fno_ckpt), device)
    print(f"FNO done in {time.time() - t0:.1f}s")

    t0 = time.time()
    mgn_pred = infer_mgn(records, cond, Path(args.mgn_ckpt), device)
    print(f"MGN done in {time.time() - t0:.1f}s")

    t0 = time.time()
    gino_pred = infer_gino(records, cond, Path(args.gino_ckpt), device)
    print(f"GINO done in {time.time() - t0:.1f}s")

    t0 = time.time()
    rnn_pred = infer_rnn(prepared["grids"], prepared["node_x"], prepared["node_y"], Path(args.rnn_ckpt), device)
    print(f"RNN done in {time.time() - t0:.1f}s")

    symm_mgn_pred = None
    if args.symm_mgn_ckpt:
        t0 = time.time()
        symm_mgn_pred = infer_symm_mgn(records, cond, Path(args.symm_mgn_ckpt), device)
        print(f"SymMGN done in {time.time() - t0:.1f}s" if symm_mgn_pred else "SymMGN skipped")

    out_path = Path(args.out)
    save_results_npz(out_path, args.case_idx, cond, prepared, fno_pred, mgn_pred, gino_pred, rnn_pred, symm_mgn_pred)
    print(f"Saved: {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
