"""
MGN vs FEM Comparison — Physical Units (Tesla + %)
===================================================
NormaliseTransform only normalizes input x (material_id), NOT target y (Bnorm).
Therefore y_true and y_pred are already in Tesla.
This script relabels all plots with proper T units and adds relative error (%).
"""
import os, sys, json, glob, re
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path

# ==============================
# Config
# ==============================
CACHE_DIR   = "/workspace/multiscale-pde-operators/datasets/motor_cached"
CKPT_DIR    = "/workspace/multiscale-pde-operators/lightning_logs/kme564qn/checkpoints"
TRAIN_LOG   = "/workspace/mgn_motor_train.log"
OUT_DIR     = "/workspace/mgn_fem_comparison_physical"
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUT_DIR, exist_ok=True)

# ==============================
# 1) Load best checkpoint
# ==============================
ckpt_files = sorted(glob.glob(os.path.join(CKPT_DIR, "epoch=*.ckpt")))
if not ckpt_files:
    raise FileNotFoundError(f"No checkpoint found in {CKPT_DIR}")
best_ckpt = ckpt_files[-1]
print(f"Loading checkpoint: {best_ckpt}")

sys.path.insert(0, "/workspace/multiscale-pde-operators")
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
import hydra

GlobalHydra.instance().clear()
initialize_config_dir(
    config_dir="/workspace/multiscale-pde-operators/configs",
    version_base=None,
)
cfg = compose(config_name="mgn_motor")

from multiscale_operator.data.datasets import CachedMotorDataModule
from multiscale_operator.model.trainer import MSOModule

data_module = CachedMotorDataModule(cfg, cache_dir=CACHE_DIR)
sample = data_module.get_sample()

operator = hydra.utils.instantiate(cfg.model_cfg.operator, cfg)
operator.init_shapes(sample)

ckpt = torch.load(best_ckpt, map_location=DEVICE, weights_only=False)
module = MSOModule(operator, cfg)
module.load_state_dict(ckpt["state_dict"])
module = module.to(DEVICE)
module.eval()
n_params = sum(p.numel() for p in module.parameters())
print(f"Model loaded to {DEVICE}, params={n_params:,}")

# ==============================
# 2) Inference on test set
# ==============================
test_dir = os.path.join(CACHE_DIR, "test")
test_files = sorted(glob.glob(os.path.join(test_dir, "*.pt")))
print(f"Test samples: {len(test_files)}")

transform = data_module.test_dataset.transform

all_true = []
all_pred = []
all_pos  = []
all_names = []
per_sample_metrics = []

from torch_geometric.data import Batch

with torch.no_grad():
    for i, pt_path in enumerate(test_files):
        data = torch.load(pt_path, map_location="cpu", weights_only=False)
        if transform is not None:
            data = transform(data)
        
        batch = Batch.from_data_list([data]).to(DEVICE)
        yhat = module(batch)
        
        # y is raw Bnorm in Tesla (NormaliseTransform only normalizes x, not y)
        y_true = batch.y.cpu().numpy()   # (n_nodes, 1) - Tesla
        y_pred = yhat.cpu().numpy()      # (n_nodes, 1) - Tesla
        pos = batch.pos.cpu().numpy()    # (n_nodes, 2) - mm
        
        all_true.append(y_true)
        all_pred.append(y_pred)
        all_pos.append(pos)
        all_names.append(os.path.basename(pt_path))
        
        # Metrics in Tesla
        err = y_pred - y_true
        rmse = np.sqrt(np.mean(err**2))
        rng = y_true.max() - y_true.min()
        nrmse = rmse / (rng + 1e-12)
        r2 = 1.0 - np.sum(err**2) / (np.sum((y_true - y_true.mean())**2) + 1e-12)
        mae = np.abs(err).mean()
        
        # Relative error (%) — avoid div-by-zero
        mask_nonzero = np.abs(y_true.ravel()) > 0.01  # > 10 mT
        if mask_nonzero.sum() > 0:
            rel_err_pct = np.abs(err.ravel()[mask_nonzero] / y_true.ravel()[mask_nonzero]) * 100
            median_rel_pct = float(np.median(rel_err_pct))
            mean_rel_pct = float(np.mean(rel_err_pct))
        else:
            median_rel_pct = mean_rel_pct = 0.0
        
        per_sample_metrics.append({
            "name": os.path.basename(pt_path),
            "rmse_T": float(rmse),
            "nrmse": float(nrmse),
            "r2": float(r2),
            "mae_T": float(mae),
            "n_nodes": len(y_true),
            "median_rel_err_pct": median_rel_pct,
            "mean_rel_err_pct": mean_rel_pct,
        })
        
        if (i + 1) % 30 == 0:
            print(f"  [{i+1}/{len(test_files)}] RMSE={rmse:.4f}T nRMSE={nrmse*100:.1f}% R²={r2:.4f}")

# Concatenate
all_true_cat = np.concatenate(all_true, axis=0)
all_pred_cat = np.concatenate(all_pred, axis=0)
total_err = all_pred_cat - all_true_cat
total_rmse = np.sqrt(np.mean(total_err**2))
total_rng = all_true_cat.max() - all_true_cat.min()
total_nrmse = total_rmse / (total_rng + 1e-12)
total_r2 = 1.0 - np.sum(total_err**2) / (np.sum((all_true_cat - all_true_cat.mean())**2) + 1e-12)
total_mae = np.abs(total_err).mean()
total_maxerr = float(np.abs(total_err).max())

# Global relative error
mask_g = np.abs(all_true_cat.ravel()) > 0.01
if mask_g.sum() > 0:
    global_rel_err = np.abs(total_err.ravel()[mask_g] / all_true_cat.ravel()[mask_g]) * 100
    global_median_rel = float(np.median(global_rel_err))
    global_mean_rel = float(np.mean(global_rel_err))
else:
    global_median_rel = global_mean_rel = 0.0

print(f"\n{'='*60}")
print(f"TOTAL TEST METRICS (Physical Units)")
print(f"  Samples : {len(test_files)},  Nodes: {len(all_true_cat):,}")
print(f"  RMSE    = {total_rmse:.4f} T")
print(f"  nRMSE   = {total_nrmse*100:.2f}%")
print(f"  R²      = {total_r2:.4f}")
print(f"  MAE     = {total_mae:.4f} T")
print(f"  Max Err = {total_maxerr:.4f} T")
print(f"  Relative Error (|B|>10mT): median={global_median_rel:.1f}%, mean={global_mean_rel:.1f}%")
print(f"{'='*60}\n")

# ==============================
# 3) Parse training log
# ==============================
epochs_data = []
with open(TRAIN_LOG, "r") as f:
    for line in f:
        m = re.search(r'\[Epoch\s+(\d+)/\d+\]\s+train_loss=([0-9.]+)\s+val_loss=([0-9.nan]+)\s+RMSE=([0-9.nan]+)\s+nRMSE=([0-9.nan]+)\s+lr=([0-9.]+)', line)
        if m:
            epochs_data.append({
                "epoch": int(m.group(1)),
                "train_loss": float(m.group(2)),
                "val_loss": float(m.group(3)) if m.group(3) != "nan" else None,
                "rmse": float(m.group(4)) if m.group(4) != "nan" else None,
                "nrmse": float(m.group(5)) if m.group(5) != "nan" else None,
                "lr": float(m.group(6)),
            })
print(f"Parsed {len(epochs_data)} epochs")

# ==============================
# 4) PLOT 1: Training History (same as before)
# ==============================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

ep_arr = [e["epoch"] for e in epochs_data]
tl_arr = [e["train_loss"] for e in epochs_data]
vl_arr = [e["val_loss"] for e in epochs_data if e["val_loss"] is not None]
vl_ep  = [e["epoch"] for e in epochs_data if e["val_loss"] is not None]
nrmse_arr = [e["nrmse"] for e in epochs_data if e["nrmse"] is not None]
nrmse_ep  = [e["epoch"] for e in epochs_data if e["nrmse"] is not None]
lr_arr = [e["lr"] for e in epochs_data]

axes[0].semilogy(ep_arr, tl_arr, 'b-', linewidth=1, label=f'Train (final={tl_arr[-1]:.4f})', alpha=0.8)
if vl_arr:
    axes[0].semilogy(vl_ep, vl_arr, 'r-', linewidth=1.5, label=f'Val (final={vl_arr[-1]:.4f})')
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("MSE Loss (log)")
axes[0].set_title("Training Convergence"); axes[0].legend(); axes[0].grid(True, alpha=0.3)

if nrmse_arr:
    axes[1].plot(nrmse_ep, [n*100 for n in nrmse_arr], 'g-', linewidth=1.5)
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("nRMSE (%)")
    axes[1].set_title(f"Validation nRMSE (final={nrmse_arr[-1]*100:.1f}%)")
    axes[1].grid(True, alpha=0.3)

axes[2].plot(ep_arr, lr_arr, 'k-', linewidth=1)
axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("Learning Rate")
axes[2].set_title("Cosine Annealing LR"); axes[2].grid(True, alpha=0.3)

fig.suptitle(f"MGN Motor Training (200 epochs, {n_params:,} params)", fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "01_training_history.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: 01_training_history.png")

# ==============================
# 5) PLOT 2: Scatter in Tesla
# ==============================
fig, ax = plt.subplots(figsize=(8, 8))
n_total = len(all_true_cat)
if n_total > 200000:
    idx = np.random.RandomState(42).choice(n_total, 200000, replace=False)
    x_plot = all_true_cat[idx].ravel()
    y_plot = all_pred_cat[idx].ravel()
else:
    x_plot = all_true_cat.ravel()
    y_plot = all_pred_cat.ravel()

ax.hexbin(x_plot, y_plot, gridsize=200, cmap="hot_r", mincnt=1)
mn, mx = min(x_plot.min(), y_plot.min()), max(x_plot.max(), y_plot.max())
ax.plot([mn, mx], [mn, mx], 'b--', linewidth=1, label="y=x (perfect)")
ax.set_xlabel("FEM Ground Truth — Bnorm [T]", fontsize=12)
ax.set_ylabel("MGN Prediction — Bnorm [T]", fontsize=12)
ax.set_title(
    f"MGN vs FEM — All Test Nodes ({n_total:,})\n"
    f"R²={total_r2:.4f}  RMSE={total_rmse:.4f} T  nRMSE={total_nrmse*100:.2f}%  MAE={total_mae:.4f} T",
    fontsize=11,
)
ax.legend(fontsize=11)
ax.set_aspect("equal")
ax.grid(True, alpha=0.2)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "02_scatter_tesla.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: 02_scatter_tesla.png")

# ==============================
# 6) PLOT 3: Error Histogram in Tesla + Relative Error %
# ==============================
fig, axes = plt.subplots(1, 3, figsize=(20, 5))

err_flat = total_err.ravel()

# (a) Absolute error in Tesla
axes[0].hist(err_flat, bins=200, color="steelblue", edgecolor="none", alpha=0.8)
axes[0].axvline(0, color="red", linestyle="--", linewidth=1)
axes[0].set_xlabel("Prediction Error [T]")
axes[0].set_ylabel("Count")
axes[0].set_title(f"Absolute Error Distribution\nMean={err_flat.mean():.5f} T  Std={err_flat.std():.5f} T")
axes[0].grid(True, alpha=0.3)

# (b) Per-sample nRMSE %
nrmse_per = [m["nrmse"]*100 for m in per_sample_metrics]
axes[1].hist(nrmse_per, bins=40, color="coral", edgecolor="none", alpha=0.8)
axes[1].axvline(np.median(nrmse_per), color="red", linestyle="--", linewidth=1.5,
               label=f"Median={np.median(nrmse_per):.1f}%")
axes[1].set_xlabel("Per-Sample nRMSE [%]")
axes[1].set_ylabel("Count")
axes[1].set_title(f"Per-Sample nRMSE Distribution\n({len(per_sample_metrics)} samples)")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# (c) Relative error % (node-level, |B| > 10 mT)
if mask_g.sum() > 0:
    rel_clip = np.clip(global_rel_err, 0, 200)  # clip at 200% for histogram
    axes[2].hist(rel_clip, bins=100, color="mediumpurple", edgecolor="none", alpha=0.8)
    axes[2].axvline(global_median_rel, color="red", linestyle="--", linewidth=1.5,
                   label=f"Median={global_median_rel:.1f}%")
    axes[2].axvline(global_mean_rel, color="orange", linestyle="--", linewidth=1.5,
                   label=f"Mean={global_mean_rel:.1f}%")
    axes[2].set_xlabel("Relative Error [%] (|error|/|true|×100)")
    axes[2].set_ylabel("Count")
    axes[2].set_title(f"Node-level Relative Error (|B|>10mT)\n{mask_g.sum():,} nodes")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

fig.suptitle("MGN Motor — Error Analysis (Physical Units)", fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "03_error_histogram_tesla.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: 03_error_histogram_tesla.png")

# ==============================
# 7) PLOT 4: Field Comparison — 3-panel (Tesla) + relative error (%)
# ==============================
nrmse_vals = np.array([m["nrmse"] for m in per_sample_metrics])
sorted_idx = np.argsort(nrmse_vals)
select_positions = [0, len(sorted_idx)//4, len(sorted_idx)//2,
                    3*len(sorted_idx)//4, int(0.9*len(sorted_idx)), -1]
selected = [sorted_idx[p] for p in select_positions]
labels = ["Best", "25th pct", "Median", "75th pct", "90th pct", "Worst"]

# 4 columns: FEM [T] | MGN [T] | Error [T] | Relative Error [%]
fig, axes = plt.subplots(len(selected), 4, figsize=(26, 4*len(selected)))
for row, (si, lbl) in enumerate(zip(selected, labels)):
    pos = all_pos[si]
    y_true = all_true[si].ravel()   # Tesla
    y_pred = all_pred[si].ravel()   # Tesla
    err = y_pred - y_true           # Tesla
    m = per_sample_metrics[si]
    
    x, y = pos[:, 0], pos[:, 1]
    vmin = min(y_true.min(), y_pred.min())
    vmax = max(y_true.max(), y_pred.max())
    
    # Col 0: FEM [T]
    sc0 = axes[row, 0].scatter(x, y, c=y_true, s=0.5, cmap="jet",
                                vmin=vmin, vmax=vmax,
                                edgecolors="none", rasterized=True)
    cb0 = plt.colorbar(sc0, ax=axes[row, 0], fraction=0.046, pad=0.04)
    cb0.set_label("[T]", fontsize=8)
    axes[row, 0].set_aspect("equal")
    axes[row, 0].set_title(f"FEM Bnorm ({lbl})", fontsize=10)
    
    # Col 1: MGN [T]
    sc1 = axes[row, 1].scatter(x, y, c=y_pred, s=0.5, cmap="jet",
                                vmin=vmin, vmax=vmax,
                                edgecolors="none", rasterized=True)
    cb1 = plt.colorbar(sc1, ax=axes[row, 1], fraction=0.046, pad=0.04)
    cb1.set_label("[T]", fontsize=8)
    axes[row, 1].set_aspect("equal")
    axes[row, 1].set_title(f"MGN Predicted ({lbl})", fontsize=10)
    
    # Col 2: Absolute Error [T]
    vlim = max(abs(err.min()), abs(err.max()), 1e-6)
    sc2 = axes[row, 2].scatter(x, y, c=err, s=0.5, cmap="RdBu_r",
                                vmin=-vlim, vmax=vlim,
                                edgecolors="none", rasterized=True)
    cb2 = plt.colorbar(sc2, ax=axes[row, 2], fraction=0.046, pad=0.04)
    cb2.set_label("[T]", fontsize=8)
    axes[row, 2].set_aspect("equal")
    axes[row, 2].set_title(f"Error (Pred−FEM) [T]\nnRMSE={m['nrmse']*100:.1f}%  R²={m['r2']:.4f}", fontsize=9)
    
    # Col 3: Relative Error [%]
    # |error|/|true|*100, clip for visibility
    safe_true = np.where(np.abs(y_true) > 0.01, y_true, np.nan)  # mask < 10mT
    rel_err_pct = np.abs(err / safe_true) * 100
    # Clip for colorbar
    rel_clip = np.clip(rel_err_pct, 0, 100)
    sc3 = axes[row, 3].scatter(x, y, c=rel_clip, s=0.5, cmap="YlOrRd",
                                vmin=0, vmax=50,
                                edgecolors="none", rasterized=True)
    cb3 = plt.colorbar(sc3, ax=axes[row, 3], fraction=0.046, pad=0.04)
    cb3.set_label("[%]", fontsize=8)
    axes[row, 3].set_aspect("equal")
    median_rel = m.get("median_rel_err_pct", 0)
    axes[row, 3].set_title(f"Relative Error [%]\nMedian={median_rel:.1f}% (|B|>10mT)", fontsize=9)
    
    # Row label
    axes[row, 0].set_ylabel(f"{m['name'][:30]}\n{lbl}\nRMSE={m['rmse_T']:.3f}T", fontsize=8)

fig.suptitle(
    "MGN vs FEM — Bnorm Field Comparison (Physical Units)\n"
    f"Columns: FEM [T] | MGN [T] | Error [T] | Relative Error [%]",
    fontsize=14, fontweight="bold",
)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(os.path.join(OUT_DIR, "04_field_comparison_tesla.png"), dpi=120, bbox_inches="tight")
plt.close(fig)
print("Saved: 04_field_comparison_tesla.png")

# ==============================
# 8) PLOT 5: Per-sample metrics
# ==============================
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# (a) R² sorted
r2_vals = [m["r2"] for m in per_sample_metrics]
axes[0, 0].bar(range(len(r2_vals)), sorted(r2_vals, reverse=True), color="steelblue", width=1.0)
axes[0, 0].axhline(np.median(r2_vals), color="red", linestyle="--", label=f"Median R²={np.median(r2_vals):.4f}")
axes[0, 0].set_xlabel("Test Sample (sorted)"); axes[0, 0].set_ylabel("R²")
axes[0, 0].set_title("Per-Sample R²"); axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)

# (b) nRMSE % sorted
axes[0, 1].bar(range(len(nrmse_per)), sorted(nrmse_per), color="coral", width=1.0)
axes[0, 1].axhline(np.median(nrmse_per), color="red", linestyle="--", label=f"Median={np.median(nrmse_per):.1f}%")
axes[0, 1].set_xlabel("Test Sample (sorted)"); axes[0, 1].set_ylabel("nRMSE [%]")
axes[0, 1].set_title("Per-Sample nRMSE [%]"); axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.3)

# (c) RMSE [T] sorted
rmse_vals = [m["rmse_T"] for m in per_sample_metrics]
axes[1, 0].bar(range(len(rmse_vals)), sorted(rmse_vals), color="teal", width=1.0)
axes[1, 0].axhline(np.median(rmse_vals), color="red", linestyle="--", label=f"Median={np.median(rmse_vals):.4f} T")
axes[1, 0].set_xlabel("Test Sample (sorted)"); axes[1, 0].set_ylabel("RMSE [T]")
axes[1, 0].set_title("Per-Sample RMSE [T]"); axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3)

# (d) Median Relative Error % sorted
rel_vals = [m["median_rel_err_pct"] for m in per_sample_metrics]
axes[1, 1].bar(range(len(rel_vals)), sorted(rel_vals), color="mediumpurple", width=1.0)
axes[1, 1].axhline(np.median(rel_vals), color="red", linestyle="--", label=f"Median={np.median(rel_vals):.1f}%")
axes[1, 1].set_xlabel("Test Sample (sorted)"); axes[1, 1].set_ylabel("Relative Error [%]")
axes[1, 1].set_title("Per-Sample Median Relative Error [%]\n(nodes with |B|>10mT)")
axes[1, 1].legend(); axes[1, 1].grid(True, alpha=0.3)

fig.suptitle(f"MGN Motor — Per-Sample Test Metrics ({len(per_sample_metrics)} samples)", fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "05_per_sample_metrics_tesla.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: 05_per_sample_metrics_tesla.png")

# ==============================
# 9) Save summary JSON
# ==============================
summary = {
    "model": "EncoderProcessorDecoder (MGN)",
    "params": n_params,
    "checkpoint": best_ckpt,
    "test_samples": len(test_files),
    "total_nodes": int(len(all_true_cat)),
    "units": "Tesla (T) — NormaliseTransform only normalizes input x (material_id), target y (Bnorm) remains in Tesla",
    "total_metrics": {
        "RMSE_T": float(total_rmse),
        "nRMSE_pct": float(total_nrmse * 100),
        "R2": float(total_r2),
        "MAE_T": float(total_mae),
        "MaxErr_T": float(total_maxerr),
        "Relative_Error_median_pct": global_median_rel,
        "Relative_Error_mean_pct": global_mean_rel,
    },
    "per_sample_summary": {
        "rmse_mean_T": float(np.mean(rmse_vals)),
        "rmse_median_T": float(np.median(rmse_vals)),
        "nrmse_mean_pct": float(np.mean(nrmse_per)),
        "nrmse_median_pct": float(np.median(nrmse_per)),
        "r2_mean": float(np.mean(r2_vals)),
        "r2_median": float(np.median(r2_vals)),
        "rel_err_median_pct": float(np.median(rel_vals)),
    },
    "training": {
        "epochs": len(epochs_data),
        "final_train_loss": epochs_data[-1]["train_loss"] if epochs_data else None,
        "final_val_loss": epochs_data[-1]["val_loss"] if epochs_data else None,
    },
    "per_sample_metrics": per_sample_metrics,
}

with open(os.path.join(OUT_DIR, "test_summary_physical.json"), "w") as f:
    json.dump(summary, f, indent=2)

print(f"\nSaved: test_summary_physical.json")
print(f"All outputs in: {OUT_DIR}")
print("Done!")
