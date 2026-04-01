#!/usr/bin/env python3
"""
Run inference with all 4 models on ALL timesteps of a specific DOE case.
Saves results as .npz for interactive notebook visualization.

Usage (inside Docker):
    python -u /workspace/host_data/infer_all_steps.py \
        --data-dir /workspace/host_data/doe_data \
        --case-idx 5 --grid-res 64 \
        --out /workspace/host_data/field_compare_allsteps.npz
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))

from doe_data_utils import (
    parse_h5_timeseries, scatter_elem_to_node,
    mesh_to_grid, build_gino_sample,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="/workspace/host_data/doe_data")
    parser.add_argument("--case-idx", type=int, default=5)
    parser.add_argument("--grid-res", type=int, default=64)
    parser.add_argument("--mgn-ckpt", default="/workspace/host_data/doe_meshgraphnet_ckpt.pt")
    parser.add_argument("--fno-ckpt", default="/workspace/host_data/doe_fno_ckpt.pt")
    parser.add_argument("--gino-ckpt", default="/workspace/host_data/doe_gino_ckpt.pt")
    parser.add_argument("--rnn-ckpt", default="/workspace/host_data/doe_rnn_ckpt.pt")
    parser.add_argument("--out", default="/workspace/host_data/field_compare_allsteps.npz")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- Load DOE data grouped by case ----
    data_dir = Path(args.data_dir)
    manifest_path = data_dir / "doe_manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)

    cases_data = []
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
    n_steps = len(case_recs)
    print(f"Case {ci}: {n_steps} timesteps")
    print(f"Condition: {cond}")

    GR = args.grid_res

    # ---- Pre-compute ground truth for all steps ----
    print(f"\n--- Computing ground truth for {n_steps} steps ---")
    gt_bx_grids = np.zeros((n_steps, GR, GR), dtype=np.float32)
    gt_by_grids = np.zeros((n_steps, GR, GR), dtype=np.float32)
    time_values = np.zeros(n_steps, dtype=np.float64)
    rotate_values = np.zeros(n_steps, dtype=np.float64)
    grid_x = grid_y = None

    for si in range(n_steps):
        gd = mesh_to_grid(case_recs[si], cond, grid_res=GR)
        if gd is not None:
            gt_bx_grids[si] = gd["target"][0]
            gt_by_grids[si] = gd["target"][1]
            if grid_x is None:
                grid_x = gd["grid_x"]
                grid_y = gd["grid_y"]
        time_values[si] = case_recs[si].get("time_s", 0.0)
        rotate_values[si] = case_recs[si].get("rotate_step", 0.0)

    print(f"  GT done. time range: [{time_values.min():.6f}, {time_values.max():.6f}]")

    results = {
        "case_idx": ci,
        "n_steps": n_steps,
        "condition": json.dumps(cond),
        "time_values": time_values,
        "rotate_values": rotate_values,
        "gt_bx": gt_bx_grids,
        "gt_by": gt_by_grids,
        "grid_x": grid_x,
        "grid_y": grid_y,
    }

    # ==================== FNO ====================
    if Path(args.fno_ckpt).exists():
        print("\n--- FNO inference (all steps) ---")
        t0 = time.time()
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

            fno_bx = np.zeros((n_steps, GR, GR), dtype=np.float32)
            fno_by = np.zeros((n_steps, GR, GR), dtype=np.float32)

            for si in range(n_steps):
                gd = mesh_to_grid(case_recs[si], cond, grid_res=GR)
                if gd is None:
                    continue
                x_inp = torch.from_numpy(gd["input"]).unsqueeze(0).to(device)
                x_inp = torch.nan_to_num(x_inp)
                x_norm = (x_inp - x_mean) / x_std
                with torch.no_grad():
                    y_norm = model(x_norm)
                    y_pred = y_norm * y_std + y_mean
                fno_bx[si] = y_pred[0, 0].cpu().numpy()
                fno_by[si] = y_pred[0, 1].cpu().numpy()

            results["fno_bx"] = fno_bx
            results["fno_by"] = fno_by
            print(f"  FNO done in {time.time()-t0:.1f}s")
            del model, ckpt
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  FNO error: {e}")

    # ==================== MGN ====================
    if Path(args.mgn_ckpt).exists():
        print("\n--- MGN inference (all steps) ---")
        t0 = time.time()
        try:
            from physicsnemo.models.meshgraphnet import MeshGraphNet
            from train_doe_meshgraphnet import build_graph
            from scipy.interpolate import griddata

            ckpt = torch.load(args.mgn_ckpt, map_location=device, weights_only=False)
            ad = ckpt["args"]

            xm = ckpt["x_mean"].to(device)
            xs = ckpt["x_std"].to(device)
            ym = ckpt["y_mean"].to(device)
            ys = ckpt["y_std"].to(device)
            em = ckpt["e_mean"].to(device)
            es = ckpt["e_std"].to(device)

            model = MeshGraphNet(
                input_dim_nodes=int(xm.shape[1]),
                input_dim_edges=int(em.shape[1]),
                output_dim=int(ym.shape[1]),
                processor_size=ad.get("processor_size", 15),
                hidden_dim_processor=ad.get("hidden_dim", 128),
                hidden_dim_node_encoder=ad.get("hidden_dim", 128),
                hidden_dim_edge_encoder=ad.get("hidden_dim", 128),
                hidden_dim_node_decoder=ad.get("hidden_dim", 128),
                aggregation="sum",
            ).to(device)
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()

            mgn_bx = np.zeros((n_steps, GR, GR), dtype=np.float32)
            mgn_by = np.zeros((n_steps, GR, GR), dtype=np.float32)

            for si in range(n_steps):
                g = build_graph(case_recs[si], cond)
                if g is None:
                    continue
                g.x = torch.nan_to_num(g.x)
                g.edge_attr = torch.nan_to_num(g.edge_attr)

                xn = (g.x.to(device) - xm) / xs
                en = (g.edge_attr.to(device) - em) / es

                g_dev = g.clone()
                g_dev.x = xn
                g_dev.edge_attr = en
                g_dev.edge_index = g.edge_index.to(device)

                with torch.no_grad():
                    yn = model(g_dev.x, g_dev.edge_attr, g_dev)
                    y_pred = yn * ys + ym

                # Interpolate node predictions to grid
                pred_bx = y_pred[:, 0].cpu().numpy()
                pred_by = y_pred[:, 1].cpu().numpy()
                pos = g.pos.numpy()
                mgn_bx[si] = griddata(pos, pred_bx, (grid_x, grid_y),
                                      method='linear', fill_value=0.0)
                mgn_by[si] = griddata(pos, pred_by, (grid_x, grid_y),
                                      method='linear', fill_value=0.0)

            results["mgn_bx"] = mgn_bx
            results["mgn_by"] = mgn_by
            print(f"  MGN done in {time.time()-t0:.1f}s")
            del model, ckpt
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  MGN error: {e}")

    # ==================== GINO ====================
    if Path(args.gino_ckpt).exists():
        print("\n--- GINO inference (all steps) ---")
        t0 = time.time()
        try:
            from neuralop.models import GINO
            from scipy.interpolate import griddata

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

            gino_bx = np.zeros((n_steps, GR, GR), dtype=np.float32)
            gino_by = np.zeros((n_steps, GR, GR), dtype=np.float32)

            for si in range(n_steps):
                s = build_gino_sample(case_recs[si], cond,
                                      latent_res=ad.get("latent_res", 32))
                if s is None:
                    continue
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
                # Interpolate to grid
                pos_x = case_recs[si]["pos_x"]
                pos_y = case_recs[si]["pos_y"]
                points = np.column_stack([pos_x, pos_y])
                gino_bx[si] = griddata(points, pred[:, 0], (grid_x, grid_y),
                                       method='linear', fill_value=0.0)
                gino_by[si] = griddata(points, pred[:, 1], (grid_x, grid_y),
                                       method='linear', fill_value=0.0)

            results["gino_bx"] = gino_bx
            results["gino_by"] = gino_by
            print(f"  GINO done in {time.time()-t0:.1f}s")
            del model, ckpt
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  GINO error: {e}")

    # ==================== Seq2SeqRNN ====================
    if Path(args.rnn_ckpt).exists():
        print("\n--- Seq2SeqRNN inference (all steps) ---")
        t0 = time.time()
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

            rnn_bx = np.zeros((n_steps, GR, GR), dtype=np.float32)
            rnn_by = np.zeros((n_steps, GR, GR), dtype=np.float32)

            # Pre-compute all grid frames
            all_frames = []
            for si in range(n_steps):
                gd = mesh_to_grid(case_recs[si], cond, grid_res=GR)
                if gd is not None:
                    bx_g = gd["target"][0]
                    by_g = gd["target"][1]
                    a_g = gd["input"][0]
                    j_g = gd["input"][1]
                    reg_g = gd["input"][2]
                    frame = np.stack([bx_g, by_g, a_g, j_g, reg_g], axis=0)
                    all_frames.append(frame)
                else:
                    all_frames.append(np.zeros((5, GR, GR), dtype=np.float32))

            for si in range(n_steps):
                start = max(0, si - seq_len + 1)
                seq = []
                for ti in range(start, start + seq_len):
                    idx = min(ti, n_steps - 1)
                    seq.append(all_frames[idx])
                while len(seq) < seq_len:
                    seq.insert(0, seq[0])

                x_seq = np.stack(seq, axis=1)  # (5, T, H, W)
                x = torch.from_numpy(x_seq).unsqueeze(0).float().to(device)
                x = torch.nan_to_num(x)
                x_norm = (x - inp_mean.unsqueeze(0)) / inp_std.unsqueeze(0)

                with torch.no_grad():
                    pred_full = model(x_norm)
                    pred_bxby = pred_full[:, :2]
                    pred = pred_bxby * tgt_std.unsqueeze(0) + tgt_mean.unsqueeze(0)

                rnn_bx[si] = pred[0, 0, -1].cpu().numpy()
                rnn_by[si] = pred[0, 1, -1].cpu().numpy()

            results["rnn_bx"] = rnn_bx
            results["rnn_by"] = rnn_by
            print(f"  RNN done in {time.time()-t0:.1f}s")
            del model, ckpt
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  RNN error: {e}")

    # ---- Save ----
    np.savez_compressed(args.out, **results)
    fsize = Path(args.out).stat().st_size / 1e6
    print(f"\nSaved: {args.out} ({fsize:.1f} MB)")
    print(f"Keys: {list(results.keys())}")


if __name__ == "__main__":
    main()
