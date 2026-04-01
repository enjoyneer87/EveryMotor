"""
MGN (Merantix multiscale-pde-operators) vs FEM Ground Truth Comparison Script
=============================================================================
- 테스트 셋 180개 샘플에 대해 MGN 예측과 FEM 원본을 비교
- Bnorm 필드 시각화 (FEM | MGN | Error) 3-panel
- Scatter plot (Predicted vs True)
- Error 히스토그램
- 전체 통계 (RMSE, nRMSE, R², MAE)
- 학습 이력 (train_loss, val_loss) 곡선

Usage (Docker 내부):
    python /workspace/compare_mgn_fem.py
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
OUT_DIR     = "/workspace/mgn_fem_comparison"
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUT_DIR, exist_ok=True)

# ==============================
# 1) Load best checkpoint
# ==============================
# find best ckpt (not last.ckpt)
ckpt_files = sorted(glob.glob(os.path.join(CKPT_DIR, "epoch=*.ckpt")))
if not ckpt_files:
    raise FileNotFoundError(f"No checkpoint found in {CKPT_DIR}")
best_ckpt = ckpt_files[-1]  # highest epoch = best (save_top_k=3)
print(f"Loading checkpoint: {best_ckpt}")

# Need to load model via Hydra config
sys.path.insert(0, "/workspace/multiscale-pde-operators")
from omegaconf import OmegaConf
import hydra

# Load config from train log or reconstruct
# We'll use Hydra compose
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
GlobalHydra.instance().clear()

initialize_config_dir(
    config_dir="/workspace/multiscale-pde-operators/configs",
    version_base=None,
)
cfg = compose(config_name="mgn_motor")
print("Config loaded: mgn_motor")

# Create operator + MSOModule
from multiscale_operator.data.datasets import CachedMotorDataModule
from multiscale_operator.model.trainer import MSOModule

data_module = CachedMotorDataModule(cfg, cache_dir=CACHE_DIR)
sample = data_module.get_sample()

operator = hydra.utils.instantiate(cfg.model_cfg.operator, cfg)
operator.init_shapes(sample)

# Load weights from checkpoint
ckpt = torch.load(best_ckpt, map_location=DEVICE, weights_only=False)
state_dict = ckpt["state_dict"]
module = MSOModule(operator, cfg)
module.load_state_dict(state_dict)
module = module.to(DEVICE)
module.eval()
print(f"Model loaded to {DEVICE}, params={sum(p.numel() for p in module.parameters()):,}")

# ==============================
# 2) Load test data + run inference
# ==============================
test_dir = os.path.join(CACHE_DIR, "test")
test_files = sorted(glob.glob(os.path.join(test_dir, "*.pt")))
print(f"Test samples: {len(test_files)}")

# Get transform from data_module's test dataset
transform = data_module.test_dataset.transform
print(f"Transform: {transform}")

all_true = []
all_pred = []
all_pos  = []
all_names = []
per_sample_metrics = []

from torch_geometric.data import Batch

with torch.no_grad():
    for i, pt_path in enumerate(test_files):
        data = torch.load(pt_path, map_location="cpu", weights_only=False)
        
        # Apply transform (normalisation etc) - same as training
        if transform is not None:
            data = transform(data)
        
        # Batch of 1
        batch = Batch.from_data_list([data]).to(DEVICE)
        
        # Predict
        yhat = module(batch)
        
        y_true = batch.y.cpu().numpy()      # (n_nodes, 1) - Bnorm
        y_pred = yhat.cpu().numpy()          # (n_nodes, 1)
        pos = batch.pos.cpu().numpy()        # (n_nodes, 2)
        
        all_true.append(y_true)
        all_pred.append(y_pred)
        all_pos.append(pos)
        all_names.append(os.path.basename(pt_path))
        
        # Per-sample metrics
        err = y_pred - y_true
        rmse = np.sqrt(np.mean(err**2))
        rng = y_true.max() - y_true.min()
        nrmse = rmse / (rng + 1e-12)
        r2 = 1.0 - np.sum(err**2) / (np.sum((y_true - y_true.mean())**2) + 1e-12)
        mae = np.abs(err).mean()
        
        per_sample_metrics.append({
            "name": os.path.basename(pt_path),
            "rmse": float(rmse),
            "nrmse": float(nrmse),
            "r2": float(r2),
            "mae": float(mae),
            "n_nodes": len(y_true),
        })
        
        if (i + 1) % 30 == 0:
            print(f"  [{i+1}/{len(test_files)}] RMSE={rmse:.4f} nRMSE={nrmse:.4f} R²={r2:.5f}")

# Concatenate all
all_true_cat = np.concatenate(all_true, axis=0)
all_pred_cat = np.concatenate(all_pred, axis=0)
total_err = all_pred_cat - all_true_cat
total_rmse = np.sqrt(np.mean(total_err**2))
total_rng = all_true_cat.max() - all_true_cat.min()
total_nrmse = total_rmse / (total_rng + 1e-12)
total_r2 = 1.0 - np.sum(total_err**2) / (np.sum((all_true_cat - all_true_cat.mean())**2) + 1e-12)
total_mae = np.abs(total_err).mean()

print(f"\n{'='*60}")
print(f"TOTAL TEST METRICS ({len(test_files)} samples, {len(all_true_cat)} nodes)")
print(f"  RMSE  = {total_rmse:.6f}")
print(f"  nRMSE = {total_nrmse:.6f} ({total_nrmse*100:.2f}%)")
print(f"  R²    = {total_r2:.6f}")
print(f"  MAE   = {total_mae:.6f}")
print(f"  MaxErr= {np.abs(total_err).max():.6f}")
print(f"{'='*60}\n")

# ==============================
# 3) Parse training history from log
# ==============================
epochs_data = []
with open(TRAIN_LOG, "r") as f:
    for line in f:
        m = re.search(r'\[Epoch\s+(\d+)/\d+\]\s+train_loss=([0-9.]+)\s+val_loss=([0-9.nan]+)\s+RMSE=([0-9.nan]+)\s+nRMSE=([0-9.nan]+)\s+lr=([0-9.]+)', line)
        if m:
            ep = int(m.group(1))
            tl = float(m.group(2))
            vl = m.group(3)
            rmse_v = m.group(4)
            nrmse_v = m.group(5)
            lr = float(m.group(6))
            epochs_data.append({
                "epoch": ep,
                "train_loss": tl,
                "val_loss": float(vl) if vl != "nan" else None,
                "rmse": float(rmse_v) if rmse_v != "nan" else None,
                "nrmse": float(nrmse_v) if nrmse_v != "nan" else None,
                "lr": lr,
            })

print(f"Parsed {len(epochs_data)} epochs from log")

# ==============================
# 4) PLOT 1: Training History
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
axes[2].set_title("Cosine Annealing LR Schedule"); axes[2].grid(True, alpha=0.3)

fig.suptitle("MGN Motor Training (200 epochs, EncoderProcessorDecoder 721K params)", fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "01_training_history.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: 01_training_history.png")

# ==============================
# 5) PLOT 2: Scatter (Predicted vs True) - All test data
# ==============================
fig, ax = plt.subplots(figsize=(8, 8))
# Subsample if too many points
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
ax.set_xlabel("FEM Ground Truth (Bnorm, normalized)", fontsize=12)
ax.set_ylabel("MGN Prediction (Bnorm, normalized)", fontsize=12)
ax.set_title(f"MGN vs FEM — All Test Nodes\nR²={total_r2:.5f}  RMSE={total_rmse:.4f}  nRMSE={total_nrmse*100:.2f}%", fontsize=12)
ax.legend(fontsize=11)
ax.set_aspect("equal")
ax.grid(True, alpha=0.2)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "02_scatter_all_test.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: 02_scatter_all_test.png")

# ==============================
# 6) PLOT 3: Error Histogram
# ==============================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
err_flat = total_err.ravel()

axes[0].hist(err_flat, bins=200, color="steelblue", edgecolor="none", alpha=0.8)
axes[0].axvline(0, color="red", linestyle="--", linewidth=1)
axes[0].set_xlabel("Prediction Error (Pred - True)")
axes[0].set_ylabel("Count")
axes[0].set_title(f"Error Distribution\nMean={err_flat.mean():.5f}  Std={err_flat.std():.5f}")
axes[0].grid(True, alpha=0.3)

# Per-sample nRMSE distribution
nrmse_per = [m["nrmse"]*100 for m in per_sample_metrics]
axes[1].hist(nrmse_per, bins=40, color="coral", edgecolor="none", alpha=0.8)
axes[1].axvline(np.median(nrmse_per), color="red", linestyle="--", linewidth=1.5, label=f"Median={np.median(nrmse_per):.1f}%")
axes[1].set_xlabel("Per-Sample nRMSE (%)")
axes[1].set_ylabel("Count")
axes[1].set_title(f"Per-Sample nRMSE Distribution ({len(per_sample_metrics)} test samples)")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "03_error_histogram.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: 03_error_histogram.png")

# ==============================
# 7) PLOT 4: Field Comparison (selected samples)
# ==============================
# Pick 6 representative test samples (best, worst, percentiles)
nrmse_vals = np.array([m["nrmse"] for m in per_sample_metrics])
sorted_idx = np.argsort(nrmse_vals)

# best, 25th pct, median, 75th pct, 90th pct, worst
select_positions = [0, len(sorted_idx)//4, len(sorted_idx)//2, 
                    3*len(sorted_idx)//4, int(0.9*len(sorted_idx)), -1]
selected = [sorted_idx[p] for p in select_positions]
labels = ["Best", "25th pct", "Median", "75th pct", "90th pct", "Worst"]

fig, axes = plt.subplots(len(selected), 3, figsize=(20, 4*len(selected)))
for row, (si, lbl) in enumerate(zip(selected, labels)):
    pos = all_pos[si]
    y_true = all_true[si].ravel()
    y_pred = all_pred[si].ravel()
    err = y_pred - y_true
    m = per_sample_metrics[si]
    
    x, y = pos[:, 0], pos[:, 1]
    vmin = min(y_true.min(), y_pred.min())
    vmax = max(y_true.max(), y_pred.max())
    
    # FEM
    sc0 = axes[row, 0].scatter(x, y, c=y_true, s=0.5, cmap="jet", vmin=vmin, vmax=vmax, edgecolors="none", rasterized=True)
    plt.colorbar(sc0, ax=axes[row, 0], fraction=0.046, pad=0.04)
    axes[row, 0].set_aspect("equal"); axes[row, 0].set_title(f"FEM ({lbl})")
    
    # MGN
    sc1 = axes[row, 1].scatter(x, y, c=y_pred, s=0.5, cmap="jet", vmin=vmin, vmax=vmax, edgecolors="none", rasterized=True)
    plt.colorbar(sc1, ax=axes[row, 1], fraction=0.046, pad=0.04)
    axes[row, 1].set_aspect("equal"); axes[row, 1].set_title(f"MGN Pred ({lbl})")
    
    # Error
    vlim = max(abs(err.min()), abs(err.max()), 1e-6)
    sc2 = axes[row, 2].scatter(x, y, c=err, s=0.5, cmap="RdBu_r", vmin=-vlim, vmax=vlim, edgecolors="none", rasterized=True)
    plt.colorbar(sc2, ax=axes[row, 2], fraction=0.046, pad=0.04)
    axes[row, 2].set_aspect("equal")
    axes[row, 2].set_title(f"Error  nRMSE={m['nrmse']*100:.1f}%  R²={m['r2']:.4f}")
    
    # Label row
    axes[row, 0].set_ylabel(f"{m['name'][:30]}\n{lbl}", fontsize=8)

fig.suptitle("MGN vs FEM Field Comparison (Bnorm, normalized) — Test Set Percentiles", fontsize=14, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(os.path.join(OUT_DIR, "04_field_comparison.png"), dpi=120, bbox_inches="tight")
plt.close(fig)
print("Saved: 04_field_comparison.png")

# ==============================
# 8) PLOT 5: Per-sample R² and nRMSE
# ==============================
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

r2_vals = [m["r2"] for m in per_sample_metrics]
axes[0].bar(range(len(r2_vals)), sorted(r2_vals, reverse=True), color="steelblue", width=1.0)
axes[0].axhline(np.median(r2_vals), color="red", linestyle="--", label=f"Median R²={np.median(r2_vals):.4f}")
axes[0].set_xlabel("Test Sample (sorted by R²)")
axes[0].set_ylabel("R²")
axes[0].set_title("Per-Sample R² (sorted)")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].bar(range(len(nrmse_per)), sorted(nrmse_per), color="coral", width=1.0)
axes[1].axhline(np.median(nrmse_per), color="red", linestyle="--", label=f"Median nRMSE={np.median(nrmse_per):.1f}%")
axes[1].set_xlabel("Test Sample (sorted by nRMSE)")
axes[1].set_ylabel("nRMSE (%)")
axes[1].set_title("Per-Sample nRMSE (sorted)")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

fig.suptitle(f"MGN Motor Test Performance — {len(per_sample_metrics)} samples", fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "05_per_sample_metrics.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: 05_per_sample_metrics.png")

# ==============================
# 9) Save summary JSON
# ==============================
summary = {
    "model": "EncoderProcessorDecoder (MGN)",
    "params": sum(p.numel() for p in module.parameters()),
    "checkpoint": best_ckpt,
    "test_samples": len(test_files),
    "total_nodes": int(len(all_true_cat)),
    "total_metrics": {
        "RMSE": float(total_rmse),
        "nRMSE": float(total_nrmse),
        "nRMSE_pct": float(total_nrmse * 100),
        "R2": float(total_r2),
        "MAE": float(total_mae),
        "MaxErr": float(np.abs(total_err).max()),
    },
    "per_sample_summary": {
        "rmse_mean": float(np.mean([m["rmse"] for m in per_sample_metrics])),
        "rmse_std": float(np.std([m["rmse"] for m in per_sample_metrics])),
        "nrmse_mean_pct": float(np.mean(nrmse_per)),
        "nrmse_median_pct": float(np.median(nrmse_per)),
        "r2_mean": float(np.mean(r2_vals)),
        "r2_median": float(np.median(r2_vals)),
    },
    "training": {
        "epochs": len(epochs_data),
        "final_train_loss": epochs_data[-1]["train_loss"] if epochs_data else None,
        "final_val_loss": epochs_data[-1]["val_loss"] if epochs_data else None,
    },
    "per_sample_metrics": per_sample_metrics,
}

with open(os.path.join(OUT_DIR, "test_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nSaved: test_summary.json")
print(f"\nAll outputs in: {OUT_DIR}")
print("Done!")
