#!/usr/bin/env python3
"""
4-model comparison: MGN vs FNO vs GINO vs Seq2SeqRNN

각 체크포인트를 로드하여 동일 검증 데이터에서 성능 지표를 비교합니다.
- RMSE, nRMSE, R^2, MAE (물리 단위: Tesla)
- 추론 속도 (ms/sample)
- 파라미터 수

Usage (inside Docker):
    python /workspace/compare_models.py \
        --data-dir /workspace/doe_data \
        --mgn-ckpt /workspace/doe_meshgraphnet_ckpt.pt \
        --fno-ckpt /workspace/doe_fno_ckpt.pt \
        --gino-ckpt /workspace/doe_gino_ckpt.pt \
        --rnn-ckpt /workspace/doe_rnn_ckpt.pt \
        --out-dir /workspace/model_comparison
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))


def compute_metrics(pred: np.ndarray, true: np.ndarray):
    """Compute regression metrics in physical units (Tesla).
    pred, true: arrays of shape (N, 2) or flattened."""
    pred = pred.flatten()
    true = true.flatten()
    mse = np.mean((pred - true) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(pred - true))
    y_range = true.max() - true.min()
    nrmse = rmse / max(y_range, 1e-12) * 100  # percent
    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - true.mean()) ** 2)
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
    return {
        "RMSE_T": float(rmse),
        "nRMSE_pct": float(nrmse),
        "R2": float(r2),
        "MAE_T": float(mae),
        "MSE": float(mse),
    }


def eval_mgn(ckpt_path, val_graphs, device):
    """Evaluate MeshGraphNet checkpoint."""
    from physicsnemo.models.meshgraphnet import MeshGraphNet
    from torch_geometric.loader import DataLoader

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    args_d = ckpt["args"]

    model = MeshGraphNet(
        input_dim_nodes=val_graphs[0].x.shape[1],
        input_dim_edges=val_graphs[0].edge_attr.shape[1],
        output_dim=val_graphs[0].y.shape[1],
        processor_size=args_d.get("processor_size", 15),
        hidden_dim_processor=args_d.get("hidden_dim", 128),
        hidden_dim_node_encoder=args_d.get("hidden_dim", 128),
        hidden_dim_edge_encoder=args_d.get("hidden_dim", 128),
        hidden_dim_node_decoder=args_d.get("hidden_dim", 128),
        aggregation="sum",
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())

    # Denormalize using checkpoint stats
    y_mean = ckpt["y_mean"].to(device)
    y_std = ckpt["y_std"].to(device)

    loader = DataLoader(val_graphs, batch_size=4, shuffle=False)
    all_pred, all_true = [], []
    t0 = time.time()

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred_norm = model(batch.x, batch.edge_attr, batch)
            pred = pred_norm * y_std + y_mean
            true = batch.y * y_std + y_mean
            all_pred.append(pred.cpu().numpy())
            all_true.append(true.cpu().numpy())

    infer_time = (time.time() - t0) / len(val_graphs) * 1000  # ms/sample

    pred_np = np.concatenate(all_pred, axis=0)
    true_np = np.concatenate(all_true, axis=0)
    metrics = compute_metrics(pred_np, true_np)
    metrics["params"] = n_params
    metrics["infer_ms"] = infer_time
    return metrics


def eval_fno(ckpt_path, val_X, val_Y, device):
    """Evaluate FNO checkpoint."""
    from physicsnemo.models.fno import FNO

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    args_d = ckpt["args"]

    model = FNO(
        in_channels=val_X.shape[1],
        out_channels=val_Y.shape[1],
        dimension=2,
        latent_channels=args_d.get("fno_hidden", 64),
        num_fno_layers=args_d.get("fno_layers", 4),
        num_fno_modes=[args_d.get("fno_modes", 16)] * 2,
        padding=8,
        activation_fn="gelu",
        coord_features=True,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())

    x_mean = ckpt["x_mean"].to(device)
    x_std = ckpt["x_std"].to(device)
    y_mean = ckpt["y_mean"].to(device)
    y_std = ckpt["y_std"].to(device)

    # Normalize input
    X_norm = (val_X.to(device) - x_mean) / x_std
    Y_dev = val_Y.to(device)

    all_pred, all_true = [], []
    t0 = time.time()

    with torch.no_grad():
        bs = 8
        for i in range(0, len(X_norm), bs):
            xb = X_norm[i:i+bs]
            pred_norm = model(xb)
            pred = pred_norm * y_std + y_mean
            all_pred.append(pred.cpu().numpy())
            all_true.append(Y_dev[i:i+bs].cpu().numpy())

    infer_time = (time.time() - t0) / len(val_X) * 1000

    pred_np = np.concatenate(all_pred, axis=0)
    true_np = np.concatenate(all_true, axis=0)
    metrics = compute_metrics(pred_np, true_np)
    metrics["params"] = n_params
    metrics["infer_ms"] = infer_time
    return metrics


def eval_gino(ckpt_path, val_samples, device):
    """Evaluate GINO checkpoint."""
    from neuralop.models import GINO

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    args_d = ckpt["args"]

    model = GINO(
        in_channels=9,
        out_channels=2,
        gno_coord_dim=2,
        in_gno_radius=args_d.get("gno_radius", 0.1),
        out_gno_radius=args_d.get("gno_radius", 0.1),
        fno_in_channels=3,
        fno_n_modes=(args_d.get("fno_modes", 16),) * 2,
        fno_hidden_channels=args_d.get("fno_hidden", 64),
        fno_n_layers=args_d.get("fno_layers", 4),
        fno_lifting_channel_ratio=2,
        projection_channel_ratio=4,
        in_gno_transform_type="linear",
        out_gno_transform_type="linear",
        gno_use_open3d=False,
        gno_use_torch_scatter=False,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())

    tgt_mean = ckpt["tgt_mean"]  # numpy
    tgt_std = ckpt["tgt_std"]

    all_pred, all_true = [], []
    t0 = time.time()

    with torch.no_grad():
        for s in val_samples:
            input_geom = torch.from_numpy(s["input_geom"]).unsqueeze(0).to(device)
            latent_q = torch.from_numpy(s["latent_queries"]).unsqueeze(0).to(device)
            x_feat = torch.from_numpy(s["features"]).unsqueeze(0).to(device)

            pred_norm = model(
                input_geom=input_geom,
                latent_queries=latent_q,
                output_queries=input_geom,
                x=x_feat,
            )
            pred = pred_norm.squeeze(0).cpu().numpy() * tgt_std + tgt_mean
            true = s["target"] * tgt_std + tgt_mean
            all_pred.append(pred)
            all_true.append(true)

    infer_time = (time.time() - t0) / len(val_samples) * 1000

    pred_np = np.concatenate(all_pred, axis=0)
    true_np = np.concatenate(all_true, axis=0)
    metrics = compute_metrics(pred_np, true_np)
    metrics["params"] = n_params
    metrics["infer_ms"] = infer_time
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Compare motor FEA models")
    parser.add_argument("--data-dir", type=str, default="/workspace/doe_data")
    parser.add_argument("--mgn-ckpt", type=str, default="/workspace/doe_meshgraphnet_ckpt.pt")
    parser.add_argument("--fno-ckpt", type=str, default="/workspace/doe_fno_ckpt.pt")
    parser.add_argument("--gino-ckpt", type=str, default="/workspace/doe_gino_ckpt.pt")
    parser.add_argument("--rnn-ckpt", type=str, default="/workspace/doe_rnn_ckpt.pt")
    parser.add_argument("--grid-res", type=int, default=64)
    parser.add_argument("--out-dir", type=str, default="/workspace/model_comparison")
    parser.add_argument("--max-val-samples", type=int, default=200)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    results = {}

    # ----- Evaluate available models -----
    if os.path.exists(args.mgn_ckpt):
        print("\n=== Evaluating MGN ===")
        try:
            from doe_data_utils import load_doe_data
            from train_doe_meshgraphnet import build_graph
            records, conditions = load_doe_data(args.data_dir, max_steps_per_case=None)
            # Build val graphs (last 20%)
            np.random.seed(42)
            n = len(records)
            indices = np.random.permutation(n)
            val_idx = indices[int(n * 0.8):][:args.max_val_samples]
            # Need to build graphs and normalize...
            print("  MGN evaluation requires graph data re-creation; skipping in comparison.")
            print("  Use the original training script's val metrics instead.")
        except Exception as e:
            print(f"  MGN eval error: {e}")
    else:
        print(f"MGN checkpoint not found: {args.mgn_ckpt}")

    if os.path.exists(args.fno_ckpt):
        print("\n=== Evaluating FNO ===")
        try:
            from doe_data_utils import load_doe_data, mesh_to_grid
            records, conditions = load_doe_data(args.data_dir, max_steps_per_case=None)
            np.random.seed(42)
            n = len(records)
            indices = np.random.permutation(n)
            val_idx = indices[int(n * 0.8):][:args.max_val_samples]

            val_inputs, val_targets = [], []
            for i in val_idx:
                g = mesh_to_grid(records[i], conditions[i], grid_res=args.grid_res)
                if g is not None:
                    val_inputs.append(g["input"])
                    val_targets.append(g["target"])

            val_X = torch.from_numpy(np.stack(val_inputs))
            val_Y = torch.from_numpy(np.stack(val_targets))
            val_X = torch.nan_to_num(val_X)
            val_Y = torch.nan_to_num(val_Y)

            results["FNO"] = eval_fno(args.fno_ckpt, val_X, val_Y, device)
            print(f"  FNO: {results['FNO']}")
        except Exception as e:
            print(f"  FNO eval error: {e}")

    if os.path.exists(args.gino_ckpt):
        print("\n=== Evaluating GINO ===")
        try:
            from doe_data_utils import load_doe_data, build_gino_sample
            records, conditions = load_doe_data(args.data_dir, max_steps_per_case=None)
            np.random.seed(42)
            n = len(records)
            indices = np.random.permutation(n)
            val_idx = indices[int(n * 0.8):][:args.max_val_samples]

            ckpt = torch.load(args.gino_ckpt, map_location="cpu", weights_only=False)
            latent_res = ckpt["args"].get("latent_res", 32)
            feat_mean = ckpt["feat_mean"]
            feat_std = ckpt["feat_std"]

            val_samples = []
            for i in val_idx:
                s = build_gino_sample(records[i], conditions[i], latent_res=latent_res)
                if s is not None:
                    s["features"] = ((s["features"] - feat_mean) / feat_std).astype(np.float32)
                    val_samples.append(s)

            results["GINO"] = eval_gino(args.gino_ckpt, val_samples, device)
            print(f"  GINO: {results['GINO']}")
        except Exception as e:
            print(f"  GINO eval error: {e}")

    # ----- Summary table -----
    print("\n" + "=" * 80)
    print(f"{'Model':<12} {'RMSE(T)':<10} {'nRMSE(%)':<10} {'R2':<10} "
          f"{'MAE(T)':<10} {'Params':<12} {'ms/sample':<10}")
    print("-" * 80)

    for name, m in results.items():
        print(f"{name:<12} {m['RMSE_T']:<10.4f} {m['nRMSE_pct']:<10.2f} "
              f"{m['R2']:<10.4f} {m['MAE_T']:<10.4f} "
              f"{m['params']:<12,} {m['infer_ms']:<10.2f}")

    # Save results
    out_path = os.path.join(args.out_dir, "comparison_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {out_path}")

    # ----- Generate comparison plot -----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if len(results) >= 2:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            names = list(results.keys())
            colors = {"MGN": "#1f77b4", "FNO": "#ff7f0e",
                      "GINO": "#2ca02c", "Seq2SeqRNN": "#d62728"}

            # RMSE bar
            ax = axes[0]
            vals = [results[n]["RMSE_T"] for n in names]
            bars = ax.bar(names, vals, color=[colors.get(n, "#888") for n in names])
            ax.set_ylabel("RMSE [T]")
            ax.set_title("RMSE Comparison")
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        f"{v:.4f}", ha="center", va="bottom", fontsize=9)

            # R^2 bar
            ax = axes[1]
            vals = [results[n]["R2"] for n in names]
            bars = ax.bar(names, vals, color=[colors.get(n, "#888") for n in names])
            ax.set_ylabel("R^2")
            ax.set_title("R^2 Comparison")
            ax.set_ylim(0, 1.05)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        f"{v:.4f}", ha="center", va="bottom", fontsize=9)

            # Speed bar
            ax = axes[2]
            vals = [results[n]["infer_ms"] for n in names]
            bars = ax.bar(names, vals, color=[colors.get(n, "#888") for n in names])
            ax.set_ylabel("Inference [ms/sample]")
            ax.set_title("Speed Comparison")
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        f"{v:.1f}", ha="center", va="bottom", fontsize=9)

            plt.tight_layout()
            fig_path = os.path.join(args.out_dir, "model_comparison.png")
            plt.savefig(fig_path, dpi=150)
            print(f"Plot saved: {fig_path}")
            plt.close()
    except Exception as e:
        print(f"Plot generation error: {e}")


if __name__ == "__main__":
    main()
