#!/usr/bin/env python3
"""
Generate node-wise predictions for all steps of one DOE case.
- Stores GT and model predictions on mesh nodes (Bx, By)
- FNO/RNN grid outputs are interpolated back to nodes per step

Output NPZ keys:
  node_x, node_y:       (S, N)
  gt_bx_node, gt_by_node
  mgn_bx_node, mgn_by_node
  gino_bx_node, gino_by_node
  fno_bx_node, fno_by_node
  rnn_bx_node, rnn_by_node
  time_values, rotate_values, case_idx, n_steps
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.interpolate import RegularGridInterpolator

sys.path.insert(0, str(Path(__file__).parent))

from doe_data_utils import parse_h5_timeseries, mesh_to_grid, scatter_elem_to_node, build_gino_sample


def sample_grid_to_nodes(grid_x, grid_y, values, node_x, node_y):
    """Bilinear interpolation from regular grid to node points."""
    gx = grid_x[0, :]
    gy = grid_y[:, 0]
    interp = RegularGridInterpolator((gy, gx), values, method="linear", bounds_error=False, fill_value=0.0)
    pts = np.column_stack([node_y, node_x])
    return interp(pts).astype(np.float32)


def load_case_records(data_dir: Path, case_idx: int):
    manifest = json.load(open(data_dir / "doe_manifest.json"))
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
            pass

    if not records:
        raise RuntimeError("No records found for case")

    return cond, records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/workspace/host_data/doe_data")
    ap.add_argument("--case-idx", type=int, default=5)
    ap.add_argument("--grid-res", type=int, default=64)
    ap.add_argument("--mgn-ckpt", default="/workspace/host_data/doe_meshgraphnet_ckpt.pt")
    ap.add_argument("--fno-ckpt", default="/workspace/host_data/doe_fno_ckpt.pt")
    ap.add_argument("--gino-ckpt", default="/workspace/host_data/doe_gino_ckpt.pt")
    ap.add_argument("--rnn-ckpt", default="/workspace/host_data/doe_rnn_ckpt.pt")
    ap.add_argument("--out", default="/workspace/host_data/field_compare_nodes_allsteps.npz")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    data_dir = Path(args.data_dir)
    cond, records = load_case_records(data_dir, args.case_idx)

    S = len(records)
    N = records[0]["_n"]
    print(f"case={args.case_idx}, steps={S}, nodes={N}")

    node_x = np.zeros((S, N), dtype=np.float32)
    node_y = np.zeros((S, N), dtype=np.float32)
    gt_bx = np.zeros((S, N), dtype=np.float32)
    gt_by = np.zeros((S, N), dtype=np.float32)
    gt_a = np.zeros((S, N), dtype=np.float32)
    gt_j = np.zeros((S, N), dtype=np.float32)
    time_values = np.zeros(S, dtype=np.float64)
    rotate_values = np.zeros(S, dtype=np.float64)

    # Precompute step-wise grid inputs/targets for FNO/RNN and interpolation coordinates
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
        grids.append(mesh_to_grid(rec, cond, grid_res=args.grid_res))

    # Allocate predictions
    fno_bx = np.zeros((S, N), dtype=np.float32)
    fno_by = np.zeros((S, N), dtype=np.float32)
    fno_a = np.zeros((S, N), dtype=np.float32)
    fno_j = np.zeros((S, N), dtype=np.float32)
    mgn_bx = np.zeros((S, N), dtype=np.float32)
    mgn_by = np.zeros((S, N), dtype=np.float32)
    mgn_a  = np.zeros((S, N), dtype=np.float32)
    mgn_j  = np.zeros((S, N), dtype=np.float32)
    gino_bx = np.zeros((S, N), dtype=np.float32)
    gino_by = np.zeros((S, N), dtype=np.float32)
    gino_a = np.zeros((S, N), dtype=np.float32)
    gino_j = np.zeros((S, N), dtype=np.float32)
    rnn_bx = np.zeros((S, N), dtype=np.float32)
    rnn_by = np.zeros((S, N), dtype=np.float32)
    rnn_a = np.zeros((S, N), dtype=np.float32)
    rnn_j = np.zeros((S, N), dtype=np.float32)

    # ---------------- FNO ----------------
    t0 = time.time()
    from physicsnemo.models.fno import FNO
    ckpt_fno = torch.load(args.fno_ckpt, map_location=device, weights_only=False)
    ad = ckpt_fno["args"]
    model_fno = FNO(
        in_channels=7,
        out_channels=4,
        dimension=2,
        latent_channels=ad.get("fno_hidden", 64),
        num_fno_layers=ad.get("fno_layers", 4),
        num_fno_modes=[ad.get("fno_modes", 16)] * 2,
        padding=8,
        activation_fn="gelu",
        coord_features=True,
    ).to(device)
    model_fno.load_state_dict(ckpt_fno["model_state_dict"])
    model_fno.eval()
    xm, xs = ckpt_fno["x_mean"].to(device), ckpt_fno["x_std"].to(device)
    ym, ys = ckpt_fno["y_mean"].to(device), ckpt_fno["y_std"].to(device)

    with torch.no_grad():
        for si, gd in enumerate(grids):
            x = torch.from_numpy(gd["input"]).unsqueeze(0).to(device)
            x = torch.nan_to_num(x)
            y_norm = model_fno((x - xm) / xs)
            y = y_norm * ys + ym
            bx_g = y[0, 0].cpu().numpy()
            by_g = y[0, 1].cpu().numpy()
            a_g = y[0, 2].cpu().numpy() if y.shape[1] > 2 else np.zeros_like(bx_g)
            j_g = y[0, 3].cpu().numpy() if y.shape[1] > 3 else np.zeros_like(bx_g)

            fno_bx[si] = sample_grid_to_nodes(gd["grid_x"], gd["grid_y"], bx_g, node_x[si], node_y[si])
            fno_by[si] = sample_grid_to_nodes(gd["grid_x"], gd["grid_y"], by_g, node_x[si], node_y[si])
            fno_a[si] = sample_grid_to_nodes(gd["grid_x"], gd["grid_y"], a_g, node_x[si], node_y[si])
            fno_j[si] = sample_grid_to_nodes(gd["grid_x"], gd["grid_y"], j_g, node_x[si], node_y[si])
    print(f"FNO done in {time.time()-t0:.1f}s")

    # ---------------- MGN ----------------
    t0 = time.time()
    from physicsnemo.models.meshgraphnet import MeshGraphNet
    from train_doe_meshgraphnet import build_graph

    ckpt_mgn = torch.load(args.mgn_ckpt, map_location=device, weights_only=False)
    ad = ckpt_mgn.get("args", {})
    x_mean, x_std = ckpt_mgn["x_mean"].to(device), ckpt_mgn["x_std"].to(device)
    y_mean, y_std = ckpt_mgn["y_mean"].to(device), ckpt_mgn["y_std"].to(device)
    e_mean = ckpt_mgn.get("e_mean", torch.tensor([[0.0, 0.0, 0.0]])).to(device)
    e_std = ckpt_mgn.get("e_std", torch.tensor([[1.0, 1.0, 1.0]])).to(device)

    model_mgn = MeshGraphNet(
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
    model_mgn.load_state_dict(ckpt_mgn["model_state_dict"])
    model_mgn.eval()

    with torch.no_grad():
        for si, rec in enumerate(records):
            g = build_graph(rec, cond)
            g.x = torch.nan_to_num(g.x)
            g.edge_attr = torch.nan_to_num(g.edge_attr)
            
            xn = (g.x.to(device) - x_mean.to(device)) / x_std.to(device)
            en = (g.edge_attr.to(device) - e_mean) / e_std
            g_dev = g.clone()
            g_dev.x = xn
            g_dev.edge_attr = en
            g_dev.edge_index = g.edge_index.to(device)
            y = model_mgn(g_dev.x, g_dev.edge_attr, g_dev) * y_std + y_mean
            mgn_bx[si] = y[:, 0].cpu().numpy()
            mgn_by[si] = y[:, 1].cpu().numpy()
            if y.shape[1] >= 4:
                mgn_a[si] = y[:, 2].cpu().numpy()
                mgn_j[si] = y[:, 3].cpu().numpy()
    print(f"MGN done in {time.time()-t0:.1f}s")

    # ---------------- GINO ----------------
    t0 = time.time()
    from neuralop.models import GINO

    ckpt_gino = torch.load(args.gino_ckpt, map_location=device, weights_only=False)
    ad = ckpt_gino["args"]
    model_gino = GINO(
        in_channels=7,
        out_channels=4,
        gno_coord_dim=2,
        in_gno_radius=ad.get("gno_radius", 0.1),
        out_gno_radius=ad.get("gno_radius", 0.1),
        fno_in_channels=ad.get("fno_in_channels", 7),
        fno_n_modes=(ad.get("fno_modes", 16),) * 2,
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
    model_gino.load_state_dict(ckpt_gino["model_state_dict"])
    model_gino.eval()

    feat_mean, feat_std = ckpt_gino["feat_mean"], ckpt_gino["feat_std"]
    tgt_mean, tgt_std = ckpt_gino["tgt_mean"], ckpt_gino["tgt_std"]

    with torch.no_grad():
        for si, rec in enumerate(records):
            s = build_gino_sample(rec, cond, latent_res=ad.get("latent_res", 32))
            s["features"] = ((s["features"] - feat_mean) / feat_std).astype(np.float32)
            input_geom = torch.from_numpy(s["input_geom"]).unsqueeze(0).to(device)
            latent_q = torch.from_numpy(s["latent_queries"]).unsqueeze(0).to(device)
            x_feat = torch.from_numpy(s["features"]).unsqueeze(0).to(device)
            pred_norm = model_gino(
                input_geom=input_geom,
                latent_queries=latent_q,
                output_queries=input_geom,
                x=x_feat,
            )
            pred = pred_norm.squeeze(0).cpu().numpy() * tgt_std + tgt_mean
            gino_bx[si] = pred[:, 0]
            gino_by[si] = pred[:, 1]
            if pred.shape[1] > 2:
                gino_a[si] = pred[:, 2]
                gino_j[si] = pred[:, 3]
    model_rnn = Seq2SeqRNN(
        input_channels=5,
        dimension=2,
        nr_latent_channels=ad.get("hidden_channels", 64),
        nr_tsteps=seq_len,
    ).to(device)
    model_rnn.load_state_dict(ckpt_rnn["model_state_dict"])
    model_rnn.eval()

    inp_mean = torch.as_tensor(ckpt_rnn["inp_mean"], device=device)
    inp_std = torch.as_tensor(ckpt_rnn["inp_std"], device=device)
    tgt_mean = torch.as_tensor(ckpt_rnn["tgt_mean"], device=device)
    tgt_std = torch.as_tensor(ckpt_rnn["tgt_std"], device=device)

    frames = []
    for gd in grids:
        frame = np.stack([gd["target"][0], gd["target"][1], gd["input"][0], gd["input"][1], gd["input"][2]], axis=0)
        frames.append(frame)

    with torch.no_grad():
        for si in range(S):
            start = max(0, si - seq_len + 1)
            seq = []
            for ti in range(start, start + seq_len):
                idx = min(ti, S - 1)
                seq.append(frames[idx])
            while len(seq) < seq_len:
                seq.insert(0, seq[0])

            x_seq = np.stack(seq, axis=1)
            x = torch.from_numpy(x_seq).unsqueeze(0).float().to(device)
            x = torch.nan_to_num(x)
            x_norm = (x - inp_mean.unsqueeze(0)) / inp_std.unsqueeze(0)
            pred_full = model_rnn(x_norm)
            out_dim = pred_full.shape[1]
            pred = pred_full[:, :out_dim] * tgt_std.unsqueeze(0) + tgt_mean.unsqueeze(0)
            bx_g = pred[0, 0, -1].cpu().numpy()
            by_g = pred[0, 1, -1].cpu().numpy()
            a_g = pred[0, 2, -1].cpu().numpy() if out_dim > 2 else np.zeros_like(bx_g)
            j_g = pred[0, 3, -1].cpu().numpy() if out_dim > 3 else np.zeros_like(bx_g)

            gd = grids[si]
            rnn_bx[si] = sample_grid_to_nodes(gd["grid_x"], gd["grid_y"], bx_g, node_x[si], node_y[si])
            rnn_by[si] = sample_grid_to_nodes(gd["grid_x"], gd["grid_y"], by_g, node_x[si], node_y[si])
            rnn_a[si] = sample_grid_to_nodes(gd["grid_x"], gd["grid_y"], a_g, node_x[si], node_y[si])
            rnn_j[si] = sample_grid_to_nodes(gd["grid_x"], gd["grid_y"], j_g, node_x[si], node_y[si])
    print(f"RNN done in {time.time()-t0:.1f}s")

    out_path = Path(args.out)
    np.savez_compressed(
        out_path,
        case_idx=np.int32(args.case_idx),
        n_steps=np.int32(S),
        n_nodes=np.int32(N),
        condition=json.dumps(cond),
        time_values=time_values,
        rotate_values=rotate_values,
        node_x=node_x,
        node_y=node_y,
        gt_bx_node=gt_bx,
        gt_by_node=gt_by,
        gt_a_node=gt_a,
        gt_j_node=gt_j,
        fno_bx_node=fno_bx,
        fno_by_node=fno_by,
        fno_a_node=fno_a,
        fno_j_node=fno_j,
        mgn_bx_node=mgn_bx,
        mgn_by_node=mgn_by,
        mgn_a_node=mgn_a,
        mgn_j_node=mgn_j,
        gino_bx_node=gino_bx,
        gino_by_node=gino_by,
        gin_a_node=gino_a,
        gin_j_node=gino_j,
        rnn_bx_node=rnn_bx,
        rnn_by_node=rnn_by,
        rnn_a_node=rnn_a,
        rnn_j_node=rnn_j,
    )

    print(f"Saved: {out_path} ({out_path.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
