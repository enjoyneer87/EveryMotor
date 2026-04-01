#!/usr/bin/env python3
"""
Run inference with all 4 models on a specific DOE case/timestep and save
ground truth + predictions for B-field comparison plotting.

Usage (inside Docker):
    python -u /workspace/host_data/infer_field_compare.py \
        --data-dir /workspace/host_data/doe_data \
        --case-idx 5 --step-idx 20 \
        --out /workspace/host_data/field_compare_results.npz

Outputs: .npz file with keys per model containing:
  - gt_bx, gt_by:       ground truth on nodes
  - pos_x, pos_y:       node positions
  - grid_gt_bx, grid_gt_by: ground truth on 64x64 grid
  - {model}_grid_bx, {model}_grid_by: FNO/RNN predictions on grid
  - {model}_node_bx, {model}_node_by: MGN/GINO predictions on nodes
  - grid_x, grid_y:     meshgrid coordinates
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))

from doe_data_utils import (
    load_doe_data, scatter_elem_to_node, mesh_to_grid, build_gino_sample,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="/workspace/host_data/doe_data")
    parser.add_argument("--case-idx", type=int, default=5,
                        help="DOE case index (0-based among sorted cases)")
    parser.add_argument("--step-idx", type=int, default=20,
                        help="Timestep index within the case")
    parser.add_argument("--grid-res", type=int, default=64)
    parser.add_argument("--mgn-ckpt", default="/workspace/host_data/doe_meshgraphnet_ckpt.pt")
    parser.add_argument("--fno-ckpt", default="/workspace/host_data/doe_fno_ckpt.pt")
    parser.add_argument("--gino-ckpt", default="/workspace/host_data/doe_gino_ckpt.pt")
    parser.add_argument("--rnn-ckpt", default="/workspace/host_data/doe_rnn_ckpt.pt")
    parser.add_argument("--out", default="/workspace/host_data/field_compare_results.npz")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- Load DOE data grouped by case ----
    data_dir = Path(args.data_dir)
    manifest_path = data_dir / "doe_manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)

    # Group records by case
    from doe_data_utils import parse_h5_timeseries
    cases_data = []  # list of (condition, [records...])
    for case in manifest["cases"]:
        h5_paths = case.get("h5_paths") or []
        if not h5_paths:
            continue
        condition = {}
        condition.update(case.get("geometry", {}))
        condition.update(case.get("electrical", {}))

        case_records = []
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
                recs = parse_h5_timeseries(h5_file)
                case_records.extend(recs)
            except Exception as e:
                print(f"  [WARN] parse error {h5_file}: {e}")

        if case_records:
            cases_data.append((condition, case_records))

    print(f"Loaded {len(cases_data)} cases")
    ci = min(args.case_idx, len(cases_data) - 1)
    cond, case_recs = cases_data[ci]
    si = min(args.step_idx, len(case_recs) - 1)
    rec = case_recs[si]

    print(f"Case {ci}: {cond}")
    print(f"Step {si}: time_s={rec.get('time_s',0):.6f}, "
          f"rotate_step={rec.get('rotate_step',0):.1f}")
    print(f"Nodes: {rec['_n']}")

    # ---- Ground truth on nodes ----
    node_bx, node_by, node_a, node_j = scatter_elem_to_node(rec)
    pos_x = rec["pos_x"]
    pos_y = rec["pos_y"]

    # ---- Ground truth on grid ----
    grid_data = mesh_to_grid(rec, cond, grid_res=args.grid_res)
    grid_gt_bx = grid_data["target"][0]  # (H, W)
    grid_gt_by = grid_data["target"][1]
    grid_x = grid_data["grid_x"]
    grid_y = grid_data["grid_y"]

    results = {
        "gt_node_bx": node_bx, "gt_node_by": node_by,
        "pos_x": pos_x, "pos_y": pos_y,
        "grid_gt_bx": grid_gt_bx, "grid_gt_by": grid_gt_by,
        "grid_x": grid_x, "grid_y": grid_y,
        "case_idx": ci, "step_idx": si,
        "condition": json.dumps(cond),
    }

    # ==================== FNO ====================
    if Path(args.fno_ckpt).exists():
        print("\n--- FNO inference ---")
        try:
            from physicsnemo.models.fno import FNO
            ckpt = torch.load(args.fno_ckpt, map_location=device, weights_only=False)
            ad = ckpt["args"]
            model = FNO(
                in_channels=9, out_channels=2, dimension=2,
                latent_channels=ad.get("fno_hidden", 64),
                num_fno_layers=ad.get("fno_layers", 4),
                num_fno_modes=[ad.get("fno_modes", 16)] * 2,
                padding=8, activation_fn="gelu", coord_features=True,
            ).to(device)
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()

            x_mean = ckpt["x_mean"].to(device)
            x_std = ckpt["x_std"].to(device)
            y_mean = ckpt["y_mean"].to(device)
            y_std = ckpt["y_std"].to(device)

            x_inp = torch.from_numpy(grid_data["input"]).unsqueeze(0).to(device)
            x_inp = torch.nan_to_num(x_inp)
            x_norm = (x_inp - x_mean) / x_std

            with torch.no_grad():
                y_norm = model(x_norm)
                y_pred = y_norm * y_std + y_mean

            fno_bx = y_pred[0, 0].cpu().numpy()
            fno_by = y_pred[0, 1].cpu().numpy()
            results["fno_grid_bx"] = fno_bx
            results["fno_grid_by"] = fno_by
            print(f"  FNO done: Bx range [{fno_bx.min():.4f}, {fno_bx.max():.4f}]")
            del model, ckpt
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  FNO error: {e}")

    # ==================== MGN ====================
    if Path(args.mgn_ckpt).exists():
        print("\n--- MGN inference ---")
        try:
            from physicsnemo.models.meshgraphnet import MeshGraphNet
            from train_doe_meshgraphnet import build_graph

            ckpt = torch.load(args.mgn_ckpt, map_location=device, weights_only=False)
            ad = ckpt["args"]

            x_mean = ckpt["x_mean"].to(device)
            x_std = ckpt["x_std"].to(device)
            y_mean = ckpt["y_mean"].to(device)
            y_std = ckpt["y_std"].to(device)
            e_mean = ckpt["e_mean"].to(device)
            e_std = ckpt["e_std"].to(device)

            model = MeshGraphNet(
                input_dim_nodes=int(x_mean.shape[1]),
                input_dim_edges=int(e_mean.shape[1]),
                output_dim=int(y_mean.shape[1]),
                processor_size=ad.get("processor_size", 15),
                hidden_dim_processor=ad.get("hidden_dim", 128),
                hidden_dim_node_encoder=ad.get("hidden_dim", 128),
                hidden_dim_edge_encoder=ad.get("hidden_dim", 128),
                hidden_dim_node_decoder=ad.get("hidden_dim", 128),
                aggregation="sum",
            ).to(device)
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()

            g = build_graph(rec, cond)
            g.x = torch.nan_to_num(g.x)
            g.edge_attr = torch.nan_to_num(g.edge_attr)

            xn = (g.x.to(device) - x_mean) / x_std
            en = (g.edge_attr.to(device) - e_mean) / e_std

            g_dev = g.clone()
            g_dev.x = xn
            g_dev.edge_attr = en
            g_dev.edge_index = g.edge_index.to(device)

            with torch.no_grad():
                yn = model(g_dev.x, g_dev.edge_attr, g_dev)
                y_pred = yn * y_std + y_mean

            mgn_bx = y_pred[:, 0].cpu().numpy()
            mgn_by = y_pred[:, 1].cpu().numpy()
            results["mgn_node_bx"] = mgn_bx
            results["mgn_node_by"] = mgn_by
            print(f"  MGN done: Bx range [{mgn_bx.min():.4f}, {mgn_bx.max():.4f}]")
            del model, ckpt
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  MGN error: {e}")

    # ==================== GINO ====================
    if Path(args.gino_ckpt).exists():
        print("\n--- GINO inference ---")
        try:
            from neuralop.models import GINO

            ckpt = torch.load(args.gino_ckpt, map_location=device, weights_only=False)
            ad = ckpt["args"]

            model = GINO(
                in_channels=9, out_channels=2, gno_coord_dim=2,
                in_gno_radius=ad.get("gno_radius", 0.1),
                out_gno_radius=ad.get("gno_radius", 0.1),
                fno_in_channels=ad.get("fno_in_channels", 9),
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
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()

            feat_mean = ckpt["feat_mean"]
            feat_std = ckpt["feat_std"]
            tgt_mean = ckpt["tgt_mean"]
            tgt_std = ckpt["tgt_std"]

            s = build_gino_sample(rec, cond,
                                  latent_res=ad.get("latent_res", 32))
            s["features"] = ((s["features"] - feat_mean) / feat_std).astype(np.float32)

            input_geom = torch.from_numpy(s["input_geom"]).unsqueeze(0).to(device)
            latent_q = torch.from_numpy(s["latent_queries"]).unsqueeze(0).to(device)
            x_feat = torch.from_numpy(s["features"]).unsqueeze(0).to(device)

            with torch.no_grad():
                pred_norm = model(
                    input_geom=input_geom,
                    latent_queries=latent_q,
                    output_queries=input_geom,
                    x=x_feat,
                )
            pred = pred_norm.squeeze(0).cpu().numpy() * tgt_std + tgt_mean
            results["gino_node_bx"] = pred[:, 0]
            results["gino_node_by"] = pred[:, 1]
            print(f"  GINO done: Bx range [{pred[:,0].min():.4f}, {pred[:,0].max():.4f}]")
            del model, ckpt
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  GINO error: {e}")

    # ==================== Seq2SeqRNN ====================
    if Path(args.rnn_ckpt).exists():
        print("\n--- Seq2SeqRNN inference ---")
        try:
            from physicsnemo.models.rnn import Seq2SeqRNN

            ckpt = torch.load(args.rnn_ckpt, map_location=device, weights_only=False)
            ad = ckpt["args"]
            seq_len = ad.get("seq_out", 4)

            model = Seq2SeqRNN(
                input_channels=5, dimension=2,
                nr_latent_channels=ad.get("hidden_channels", 64),
                nr_tsteps=seq_len,
            ).to(device)
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()

            inp_mean = torch.as_tensor(ckpt["inp_mean"], device=device)
            inp_std = torch.as_tensor(ckpt["inp_std"], device=device)
            tgt_mean = torch.as_tensor(ckpt["tgt_mean"], device=device)
            tgt_std = torch.as_tensor(ckpt["tgt_std"], device=device)

            # Build input sequence from consecutive timesteps
            start_si = max(0, si - seq_len)
            seq_grids = []
            for ti in range(start_si, min(start_si + seq_len, len(case_recs))):
                gd = mesh_to_grid(case_recs[ti], cond, grid_res=args.grid_res)
                if gd is not None:
                    # channels: [Bx, By, A, J, region]
                    bx_g = gd["target"][0]
                    by_g = gd["target"][1]
                    a_g = gd["input"][0]
                    j_g = gd["input"][1]
                    reg_g = gd["input"][2]
                    frame = np.stack([bx_g, by_g, a_g, j_g, reg_g], axis=0)
                    seq_grids.append(frame)

            # Pad if not enough timesteps
            while len(seq_grids) < seq_len:
                seq_grids.insert(0, seq_grids[0])

            x_seq = np.stack(seq_grids[:seq_len], axis=1)  # (5, T, H, W)
            x_seq = torch.from_numpy(x_seq).unsqueeze(0).float().to(device)
            x_seq = torch.nan_to_num(x_seq)

            x_norm = (x_seq - inp_mean.unsqueeze(0)) / inp_std.unsqueeze(0)

            with torch.no_grad():
                pred_full = model(x_norm)  # (1, 5, T, H, W)
                pred_bxby = pred_full[:, :2]
                pred = pred_bxby * tgt_std.unsqueeze(0) + tgt_mean.unsqueeze(0)

            # Take last timestep prediction
            rnn_bx = pred[0, 0, -1].cpu().numpy()
            rnn_by = pred[0, 1, -1].cpu().numpy()
            results["rnn_grid_bx"] = rnn_bx
            results["rnn_grid_by"] = rnn_by
            print(f"  RNN done: Bx range [{rnn_bx.min():.4f}, {rnn_bx.max():.4f}]")
            del model, ckpt
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  RNN error: {e}")

    # ---- Save results ----
    np.savez_compressed(args.out, **results)
    print(f"\nResults saved: {args.out}")
    print(f"Keys: {list(results.keys())}")


if __name__ == "__main__":
    main()
