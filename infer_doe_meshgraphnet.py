#!/usr/bin/env python3
r"""
Inference & visualisation for trained DOE MeshGraphNet.

Loads a checkpoint, parses test H5 file(s), runs prediction, and
produces comparison plots: FEA ground truth vs MeshGraphNet prediction.

Can run:
  - Inside Docker (GPU):
      python /workspace/infer_doe_meshgraphnet.py --ckpt /workspace/doe_meshgraphnet_ckpt.pt \
             --data-dir /workspace/doe_data --case-idx 5 --step-idx 20 --save-dir /workspace/plots
  - On Windows (CPU, with torch + torch_geometric installed):
      python infer_doe_meshgraphnet.py --ckpt doe_meshgraphnet_ckpt.pt \
             --data-dir D:\\KDH\\Sim_4SolverX\\DOE_TrainingData --case-idx 5 --step-idx 20

Dependencies: torch, torch_geometric, physicsnemo(Docker) or standalone MeshGraphNet, h5py, numpy, matplotlib
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.tri import Triangulation

# ---------------------------------------------------------------------------
# Re-use H5 parser + graph builder from training script
# ---------------------------------------------------------------------------
# Import from the training script if available, otherwise inline them
try:
    from train_doe_meshgraphnet import parse_h5_timeseries, build_graph
except ImportError:
    # Fallback: copy the functions (should not normally happen)
    print("[WARN] Could not import from train_doe_meshgraphnet, using inline copies")
    sys.path.insert(0, str(Path(__file__).parent))
    from train_doe_meshgraphnet import parse_h5_timeseries, build_graph


# ---------------------------------------------------------------------------
# Load checkpoint + model
# ---------------------------------------------------------------------------

def load_model_and_stats(ckpt_path: str, device: torch.device):
    """Load MeshGraphNet model + normalisation stats from checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Normalisation tensors
    x_mean = ckpt["x_mean"].to(device)
    x_std  = ckpt["x_std"].to(device)
    y_mean = ckpt["y_mean"].to(device)
    y_std  = ckpt["y_std"].to(device)
    e_mean = ckpt["e_mean"].to(device)
    e_std  = ckpt["e_std"].to(device)

    args_saved = ckpt.get("args", {})
    input_dim_nodes = int(x_mean.shape[1]) if x_mean.dim() == 2 else 11
    input_dim_edges = int(e_mean.shape[1]) if e_mean.dim() == 2 else 3
    output_dim      = int(y_mean.shape[1]) if y_mean.dim() == 2 else 2
    hidden_dim      = args_saved.get("hidden_dim", 128)
    processor_size  = args_saved.get("processor_size", 15)

    try:
        from physicsnemo.models.meshgraphnet import MeshGraphNet
    except ImportError:
        from torch_geometric.nn import MessagePassing  # type: ignore
        raise ImportError(
            "physicsnemo.models.meshgraphnet not available. "
            "Run inside the PhysicsNeMo Docker container."
        )

    model = MeshGraphNet(
        input_dim_nodes=input_dim_nodes,
        input_dim_edges=input_dim_edges,
        output_dim=output_dim,
        processor_size=processor_size,
        hidden_dim_processor=hidden_dim,
        hidden_dim_node_encoder=hidden_dim,
        hidden_dim_edge_encoder=hidden_dim,
        hidden_dim_node_decoder=hidden_dim,
        aggregation="sum",
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded model ({n_params:,} params) from epoch {ckpt.get('epoch','?')}")
    print(f"  Best val MSE: {ckpt.get('val_hist',['?'])[-1] if 'val_hist' in ckpt else '?'}")

    norm = dict(x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std,
                e_mean=e_mean, e_std=e_std)
    return model, norm, ckpt


# ---------------------------------------------------------------------------
# Inference on a single graph
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_one(model, graph, norm, device) -> Tuple[np.ndarray, np.ndarray]:
    """Run inference on a single graph.

    Returns:
        pred_bxy: (N, 2) predicted [Bx, By] in *physical* units
        true_bxy: (N, 2) ground-truth [Bx, By] in *physical* units
    """
    g = graph.clone()
    # Keep original y (un-normalised) for ground truth
    true_bxy_np = graph.y.numpy().copy()

    # Sanitize
    g.x = torch.nan_to_num(g.x, 0.0, 0.0, 0.0)
    g.edge_attr = torch.nan_to_num(g.edge_attr, 0.0, 0.0, 0.0)
    g.y = torch.nan_to_num(g.y, 0.0, 0.0, 0.0)

    # Move everything to device FIRST, then normalise
    g_x = g.x.to(device)
    g_ea = g.edge_attr.to(device)
    g_ei = g.edge_index.to(device)

    xn = (g_x - norm["x_mean"]) / norm["x_std"]
    en = (g_ea - norm["e_mean"]) / norm["e_std"]

    # Build a minimal Data on device for model forward (needs edge_index)
    g_dev = g.clone()
    g_dev.x = xn
    g_dev.edge_attr = en
    g_dev.edge_index = g_ei

    # Forward
    yn_pred = model(g_dev.x, g_dev.edge_attr, g_dev)

    # Denormalise prediction back to physical units
    pred_bxy = (yn_pred * norm["y_std"] + norm["y_mean"]).cpu().numpy()

    return pred_bxy, true_bxy_np


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def plot_field_comparison(
    pos: np.ndarray,          # (N, 2) x, y
    true_bxy: np.ndarray,     # (N, 2) Bx, By
    pred_bxy: np.ndarray,     # (N, 2) Bx, By
    title_prefix: str = "",
    save_path: Optional[str] = None,
    dpi: int = 150,
):
    """Create a 3×2 figure: [Bx_true, By_true], [Bx_pred, By_pred], [Bx_err, By_err]."""
    x, y = pos[:, 0], pos[:, 1]

    # Build Delaunay triangulation for scatter→contour
    try:
        tri = Triangulation(x, y)
    except Exception:
        tri = None

    err = pred_bxy - true_bxy
    abs_err = np.abs(err)

    labels   = ["Bx", "By"]
    datasets = [
        (true_bxy, "FEA Ground Truth"),
        (pred_bxy, "MeshGraphNet Prediction"),
        (err,      "Error (Pred − FEA)"),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(14, 16))

    for col, label in enumerate(labels):
        for row, (data, row_title) in enumerate(datasets):
            ax = axes[row, col]
            vals = data[:, col]

            if row < 2:
                # Same color range for truth & prediction
                vmin = min(true_bxy[:, col].min(), pred_bxy[:, col].min())
                vmax = max(true_bxy[:, col].max(), pred_bxy[:, col].max())
            else:
                vlim = max(abs(vals.min()), abs(vals.max()), 1e-6)
                vmin, vmax = -vlim, vlim

            if tri is not None:
                tpc = ax.tripcolor(tri, vals, shading="flat", cmap="RdBu_r" if row == 2 else "jet",
                                   vmin=vmin, vmax=vmax)
            else:
                tpc = ax.scatter(x, y, c=vals, s=0.5, cmap="RdBu_r" if row == 2 else "jet",
                                 vmin=vmin, vmax=vmax)

            plt.colorbar(tpc, ax=ax, fraction=0.046, pad=0.04)
            ax.set_aspect("equal")
            ax.set_title(f"{row_title}: {label}")
            ax.set_xlabel("x [mm]")
            ax.set_ylabel("y [mm]")

    fig.suptitle(f"{title_prefix} — Bx / By comparison", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    return fig


def plot_error_histogram(
    true_bxy: np.ndarray,
    pred_bxy: np.ndarray,
    title_prefix: str = "",
    save_path: Optional[str] = None,
    dpi: int = 150,
):
    """Histogram of per-node absolute errors for Bx and By."""
    err = pred_bxy - true_bxy
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for col, label in enumerate(["Bx", "By"]):
        ax = axes[col]
        e = err[:, col]
        ax.hist(e, bins=100, edgecolor="black", alpha=0.7)
        ax.axvline(0, color="red", ls="--", lw=1)
        ax.set_xlabel(f"{label} error [T]")
        ax.set_ylabel("Count")
        ax.set_title(f"{title_prefix}: {label} error distribution\n"
                      f"MAE={np.abs(e).mean():.4f} T, RMSE={np.sqrt((e**2).mean()):.4f} T")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    return fig


def plot_scatter_pred_vs_true(
    true_bxy: np.ndarray,
    pred_bxy: np.ndarray,
    title_prefix: str = "",
    save_path: Optional[str] = None,
    dpi: int = 150,
):
    """Scatter plot of predicted vs true values for Bx and By."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for col, label in enumerate(["Bx", "By"]):
        ax = axes[col]
        t = true_bxy[:, col]
        p = pred_bxy[:, col]
        ax.scatter(t, p, s=0.3, alpha=0.3, c="steelblue")
        lim = max(abs(t).max(), abs(p).max()) * 1.05
        ax.plot([-lim, lim], [-lim, lim], "r--", lw=1, label="y=x")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.set_xlabel(f"FEA {label} [T]")
        ax.set_ylabel(f"Predicted {label} [T]")

        r2 = 1 - np.sum((t - p)**2) / (np.sum((t - t.mean())**2) + 1e-12)
        ax.set_title(f"{title_prefix}: {label}  R²={r2:.5f}")
        ax.legend()

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    return fig


def compute_metrics(true_bxy: np.ndarray, pred_bxy: np.ndarray) -> dict:
    """Compute per-component and combined error metrics."""
    err = pred_bxy - true_bxy
    metrics = {}
    for col, name in enumerate(["Bx", "By"]):
        e = err[:, col]
        t = true_bxy[:, col]
        p = pred_bxy[:, col]
        metrics[name] = {
            "MAE":  float(np.abs(e).mean()),
            "RMSE": float(np.sqrt((e**2).mean())),
            "MaxErr": float(np.abs(e).max()),
            "R2":   float(1 - np.sum(e**2) / (np.sum((t - t.mean())**2) + 1e-12)),
            "NRMSE_pct": float(np.sqrt((e**2).mean()) / (t.max() - t.min() + 1e-12) * 100),
        }

    # Combined |B| error
    b_true = np.sqrt(true_bxy[:, 0]**2 + true_bxy[:, 1]**2)
    b_pred = np.sqrt(pred_bxy[:, 0]**2 + pred_bxy[:, 1]**2)
    b_err  = b_pred - b_true
    metrics["|B|"] = {
        "MAE":  float(np.abs(b_err).mean()),
        "RMSE": float(np.sqrt((b_err**2).mean())),
        "MaxErr": float(np.abs(b_err).max()),
        "R2":   float(1 - np.sum(b_err**2) / (np.sum((b_true - b_true.mean())**2) + 1e-12)),
        "NRMSE_pct": float(np.sqrt((b_err**2).mean()) / (b_true.max() - b_true.min() + 1e-12) * 100),
    }
    return metrics


# ---------------------------------------------------------------------------
# Multi-case batch inference
# ---------------------------------------------------------------------------

def infer_all_cases(
    model, norm, manifest, data_dir: Path, device,
    max_steps_per_case: Optional[int] = None,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[dict]]:
    """Run inference on all cases in manifest.
    Returns lists of (pos, true_bxy, pred_bxy, case_info) per timestep-graph."""
    all_pos, all_true, all_pred, all_info = [], [], [], []

    for case in manifest["cases"]:
        h5_paths = case.get("h5_paths") or []
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
            except Exception:
                continue

            for si, rec in enumerate(records):
                g = build_graph(rec, condition)
                if g is None:
                    continue

                pred_bxy, true_bxy = predict_one(model, g, norm, device)
                pos = g.pos.numpy()

                all_pos.append(pos)
                all_true.append(true_bxy)
                all_pred.append(pred_bxy)
                all_info.append({
                    "case_idx": case["index"],
                    "step_idx": si,
                    "time_s":   rec.get("time_s", 0),
                    "condition": condition,
                    "h5": h5_basename,
                })

    return all_pos, all_true, all_pred, all_info


# ---------------------------------------------------------------------------
# Training history plot
# ---------------------------------------------------------------------------

def plot_training_history(ckpt: dict, save_path: Optional[str] = None, dpi: int = 150):
    """Plot train / val loss curves from checkpoint."""
    tr = ckpt.get("train_hist", [])
    va = ckpt.get("val_hist", [])
    if not tr:
        print("No training history in checkpoint")
        return None

    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = list(range(1, len(tr) + 1))
    ax.semilogy(epochs, tr, "b-", label="Train MSE", linewidth=1.5)
    ax.semilogy(epochs, va, "r-", label="Val MSE", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss (log)")
    ax.set_title("MeshGraphNet Training Convergence")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Annotate final values
    ax.annotate(f"Train: {tr[-1]:.6f}", xy=(len(tr), tr[-1]),
                xytext=(len(tr)*0.7, tr[-1]*2), fontsize=9,
                arrowprops=dict(arrowstyle="->", color="blue"), color="blue")
    ax.annotate(f"Val: {va[-1]:.6f}", xy=(len(va), va[-1]),
                xytext=(len(va)*0.7, va[-1]*3), fontsize=9,
                arrowprops=dict(arrowstyle="->", color="red"), color="red")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    return fig


# ---------------------------------------------------------------------------
# PBC boundary continuity check (Phase 1)
# ---------------------------------------------------------------------------

def plot_pbc_boundary_check(
    graph,
    pred_B: np.ndarray,
    true_B: np.ndarray,
    save_path: Optional[str] = None,
    dpi: int = 150,
):
    """Visualise predicted B at master/slave PBC boundary nodes.

    Checks that:
      1. Bx and By are continuous across the boundary (master ≈ slave values).
      2. The sign reversal (anti-periodic) is consistent with FEM ground truth.

    Args:
        graph:     torch_geometric Data with .pos, .master_idx, .slave_idx.
        pred_B:    (N, 2) predicted [Bx, By] in physical units.
        true_B:    (N, 2) FEM [Bx, By] in physical units.
        save_path: Optional file path to save the figure.
    """
    if not (hasattr(graph, "master_idx") and hasattr(graph, "slave_idx")):
        print("[plot_pbc_boundary_check] graph has no master_idx / slave_idx; skipping.")
        return None

    master_idx = graph.master_idx.numpy() if hasattr(graph.master_idx, "numpy") else np.array(graph.master_idx)
    slave_idx  = graph.slave_idx.numpy()  if hasattr(graph.slave_idx,  "numpy") else np.array(graph.slave_idx)

    if len(master_idx) == 0 or len(slave_idx) == 0:
        print("[plot_pbc_boundary_check] Empty boundary indices; skipping.")
        return None

    pos = graph.pos.numpy() if hasattr(graph.pos, "numpy") else np.array(graph.pos)

    # Sort by radius to get comparable position plots
    r_master = np.hypot(pos[master_idx, 0], pos[master_idx, 1])
    r_slave  = np.hypot(pos[slave_idx,  0], pos[slave_idx,  1])
    order_m  = np.argsort(r_master)
    order_s  = np.argsort(r_slave)

    m_idx_s = master_idx[order_m]
    s_idx_s = slave_idx[order_s]
    r_m = r_master[order_m]
    r_s = r_slave[order_s]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("PBC Boundary Check: Master vs Slave (anti-periodic symmetry)")
    comp_labels = ["Bx", "By"]
    colors = {"master_pred": "#1f77b4", "slave_pred": "#ff7f0e",
              "master_true": "#aec7e8", "slave_true": "#ffbb78"}

    for ci, comp in enumerate(comp_labels):
        ax_pred = axes[0, ci]
        ax_true = axes[1, ci]

        ax_pred.plot(r_m, pred_B[m_idx_s, ci],  color=colors["master_pred"], label="Master (pred)")
        ax_pred.plot(r_s, pred_B[s_idx_s, ci],  color=colors["slave_pred"],  label="Slave (pred)")
        ax_pred.plot(r_m, -pred_B[m_idx_s, ci], color=colors["master_pred"], ls="--", alpha=0.4, label="-Master (pred)")
        ax_pred.set_title(f"Prediction: {comp} on boundary")
        ax_pred.set_xlabel("Radius (mm)")
        ax_pred.set_ylabel(comp)
        ax_pred.legend(fontsize=8)
        ax_pred.grid(True, alpha=0.3)

        ax_true.plot(r_m, true_B[m_idx_s, ci],  color=colors["master_true"], label="Master (FEM)")
        ax_true.plot(r_s, true_B[s_idx_s, ci],  color=colors["slave_true"],  label="Slave (FEM)")
        ax_true.plot(r_m, -true_B[m_idx_s, ci], color=colors["master_true"], ls="--", alpha=0.4, label="-Master (FEM)")
        ax_true.set_title(f"FEM: {comp} on boundary")
        ax_true.set_xlabel("Radius (mm)")
        ax_true.set_ylabel(comp)
        ax_true.legend(fontsize=8)
        ax_true.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"  Saved PBC check: {save_path}")
    return fig


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Inference & viz for DOE MeshGraphNet")
    parser.add_argument("--ckpt", type=str, default="/workspace/doe_meshgraphnet_ckpt.pt")
    parser.add_argument("--data-dir", type=str, default="/workspace/doe_data")
    parser.add_argument("--case-idx", type=int, default=None,
                        help="Specific case index to visualise (None=all)")
    parser.add_argument("--step-idx", type=int, default=None,
                        help="Specific timestep within case (None=middle)")
    parser.add_argument("--save-dir", type=str, default="./plots")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--all-metrics", action="store_true",
                        help="Compute metrics over ALL cases/steps (may be slow)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- Load model ----
    model, norm, ckpt = load_model_and_stats(args.ckpt, device)

    # ---- Save dir ----
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ---- Training history plot ----
    plot_training_history(ckpt, save_path=str(save_dir / "training_history.png"))

    # ---- Load manifest ----
    data_dir = Path(args.data_dir)
    with open(data_dir / "doe_manifest.json") as f:
        manifest = json.load(f)

    # ---- Single case visualisation ----
    case_idx = args.case_idx
    if case_idx is None:
        # Pick a middle case for visualisation
        case_idx = len(manifest["cases"]) // 2

    case = None
    for c in manifest["cases"]:
        if c["index"] == case_idx:
            case = c
            break

    if case is None:
        print(f"Case {case_idx} not found in manifest. Available: "
              f"{[c['index'] for c in manifest['cases'][:10]]}...")
        sys.exit(1)

    condition = {}
    condition.update(case.get("geometry", {}))
    condition.update(case.get("electrical", {}))

    # Find H5
    h5_paths = case.get("h5_paths", [])
    h5_file = None
    for h5p in h5_paths:
        h5_basename = h5p.replace("\\", "/").split("/")[-1]
        for c in [Path(h5p),
                  data_dir / f"case_{case_idx:04d}" / "postproc" / h5_basename,
                  data_dir / h5_basename]:
            if c.exists():
                h5_file = c
                break
        if h5_file:
            break

    if h5_file is None:
        print(f"No H5 found for case {case_idx}")
        sys.exit(1)

    print(f"\nCase {case_idx}: {h5_file.name}")
    print(f"  Condition: {condition}")

    records = parse_h5_timeseries(h5_file, max_steps=args.max_steps)
    n_steps = len(records)
    step_idx = args.step_idx if args.step_idx is not None else n_steps // 2
    step_idx = min(step_idx, n_steps - 1)

    print(f"  Timestep {step_idx}/{n_steps} (t={records[step_idx].get('time_s', 0):.6f}s)")

    g = build_graph(records[step_idx], condition)
    if g is None:
        print("Failed to build graph")
        sys.exit(1)

    pred_bxy, true_bxy = predict_one(model, g, norm, device)
    pos = g.pos.numpy()

    # ---- Metrics ----
    m = compute_metrics(true_bxy, pred_bxy)
    print(f"\n  === Metrics (case {case_idx}, step {step_idx}) ===")
    for comp, vals in m.items():
        print(f"    {comp:4s}: MAE={vals['MAE']:.5f}T  RMSE={vals['RMSE']:.5f}T  "
              f"R²={vals['R2']:.5f}  NRMSE={vals['NRMSE_pct']:.2f}%  MaxErr={vals['MaxErr']:.5f}T")

    prefix = f"case{case_idx:04d}_step{step_idx:03d}"

    # ---- Plots ----
    plot_field_comparison(
        pos, true_bxy, pred_bxy,
        title_prefix=f"Case {case_idx} Step {step_idx}",
        save_path=str(save_dir / f"{prefix}_field_comparison.png"),
    )

    plot_error_histogram(
        true_bxy, pred_bxy,
        title_prefix=f"Case {case_idx} Step {step_idx}",
        save_path=str(save_dir / f"{prefix}_error_hist.png"),
    )

    plot_scatter_pred_vs_true(
        true_bxy, pred_bxy,
        title_prefix=f"Case {case_idx} Step {step_idx}",
        save_path=str(save_dir / f"{prefix}_scatter.png"),
    )

    # ---- Save single-case metrics JSON ----
    metrics_json_path = save_dir / f"{prefix}_metrics.json"
    with open(metrics_json_path, "w") as fj:
        json.dump({"case_idx": case_idx, "step_idx": step_idx, "metrics": m}, fj, indent=2)
    print(f"  Metrics JSON: {metrics_json_path}")

    # ---- Optional: all-case metrics ----
    if args.all_metrics:
        print("\nComputing metrics over ALL cases (may take a while)...", flush=True)
        all_pos, all_true, all_pred, all_info = infer_all_cases(
            model, norm, manifest, data_dir, device,
            max_steps_per_case=args.max_steps,
        )
        if all_true:
            combined_true = np.concatenate(all_true, axis=0)
            combined_pred = np.concatenate(all_pred, axis=0)
            m_all = compute_metrics(combined_true, combined_pred)
            print(f"\n  === Global Metrics ({len(all_true)} graphs, "
                  f"{combined_true.shape[0]:,} nodes total) ===", flush=True)
            for comp, vals in m_all.items():
                print(f"    {comp:4s}: MAE={vals['MAE']:.5f}T  RMSE={vals['RMSE']:.5f}T  "
                      f"R²={vals['R2']:.5f}  NRMSE={vals['NRMSE_pct']:.2f}%", flush=True)

            # Save global metrics JSON
            global_metrics_path = save_dir / "all_cases_metrics.json"
            with open(global_metrics_path, "w") as fj:
                json.dump({
                    "n_graphs": len(all_true),
                    "n_nodes_total": int(combined_true.shape[0]),
                    "metrics": m_all,
                }, fj, indent=2)
            print(f"  Global metrics JSON: {global_metrics_path}", flush=True)

            plot_scatter_pred_vs_true(
                combined_true, combined_pred,
                title_prefix="All Cases Combined",
                save_path=str(save_dir / "all_cases_scatter.png"),
            )

    print(f"\nPlots saved to: {save_dir}", flush=True)
    plt.show()


if __name__ == "__main__":
    main()
