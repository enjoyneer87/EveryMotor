#!/usr/bin/env python3
"""
Motor FEA Neural Operator Model Comparison
Docker 내에서 실행하여 결과를 /workspace/host_data/ 에 저장합니다.
"""
import torch
import numpy as np
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path("/workspace/host_data")

def count_params(state_dict):
    """Count parameters, handling nested dicts."""
    total = 0
    for v in state_dict.values():
        if isinstance(v, torch.Tensor):
            total += v.numel()
        elif isinstance(v, dict):
            total += count_params(v)
    return total

# ===== 1. Load checkpoints =====
print("=" * 60)
print("  Motor FEA Neural Operator Comparison")
print("=" * 60)

ckpt_paths = {
    "MGN": BASE / "doe_meshgraphnet_ckpt.pt",
    "FNO": BASE / "doe_fno_ckpt.pt",
    "GINO": BASE / "doe_gino_ckpt.pt",
    "Seq2SeqRNN": BASE / "doe_rnn_ckpt.pt",
}

ckpts = {}
for name, path in ckpt_paths.items():
    if path.exists():
        ckpts[name] = torch.load(path, map_location="cpu", weights_only=False)
        print(f"  OK {name}: {path.stat().st_size / 1e6:.1f} MB")
    else:
        print(f"  -- {name}: NOT FOUND")

print(f"\nLoaded {len(ckpts)} models\n")

# ===== 2. Extract histories & summary =====
histories = {}
summary = {}
colors = {"MGN": "#1f77b4", "FNO": "#ff7f0e", "GINO": "#2ca02c", "Seq2SeqRNN": "#d62728"}

for name, ckpt in ckpts.items():
    train_hist = ckpt.get("train_hist", ckpt.get("train_history", []))
    val_hist = ckpt.get("val_hist", ckpt.get("val_history", []))
    n_params = count_params(ckpt["model_state_dict"])

    if train_hist and val_hist:
        histories[name] = {"train": train_hist, "val": val_hist}
        summary[name] = {
            "parameters": n_params,
            "epochs": ckpt.get("epoch", len(train_hist)),
            "final_train_mse": train_hist[-1],
            "final_val_mse": val_hist[-1],
            "best_val_mse": min(val_hist),
            "ckpt_mb": round(ckpt_paths[name].stat().st_size / 1e6, 1),
        }

# Print summary table
print(f"{'Model':<14} {'Params':>10} {'Epochs':>6} {'Train MSE':>12} {'Val MSE':>12} {'Best Val':>12} {'Size(MB)':>8}")
print("-" * 80)
for name, s in summary.items():
    print(f"{name:<14} {s['parameters']:>10,} {s['epochs']:>6} "
          f"{s['final_train_mse']:>12.6f} {s['final_val_mse']:>12.6f} "
          f"{s['best_val_mse']:>12.6f} {s['ckpt_mb']:>8.1f}")

# ===== 3. Hyperparameters =====
print("\n")
for name, ckpt in ckpts.items():
    args_d = ckpt.get("args", {})
    model_name = ckpt.get("model_name", name)
    print(f"{'='*50}")
    print(f"  {model_name} Hyperparameters")
    print(f"{'='*50}")
    for k, v in sorted(args_d.items()):
        print(f"    {k}: {v}")
    print()

# ===== 4. Training curves =====
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for name, hist in histories.items():
    epochs = range(1, len(hist["train"]) + 1)
    c = colors.get(name, "gray")
    axes[0].plot(epochs, hist["train"], color=c, label=name, linewidth=1.5)
    axes[1].plot(epochs, hist["val"], color=c, label=name, linewidth=1.5)

for ax, title in zip(axes, ["Train Loss (MSE, normalized)", "Validation Loss (MSE, normalized)"]):
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.set_title(title)
    ax.legend()
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(str(BASE / "training_curves.png"), dpi=150, bbox_inches="tight")
print("Saved: training_curves.png")
plt.close()

# ===== 5. Bar charts =====
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

names = list(histories.keys())
best_vals = [min(histories[n]["val"]) for n in names]
final_trains = [histories[n]["train"][-1] for n in names]
n_params_list = [summary[n]["parameters"] for n in names]
bar_colors = [colors.get(n, "gray") for n in names]

# Best Val MSE
ax = axes[0]
bars = ax.bar(names, best_vals, color=bar_colors, edgecolor="black", linewidth=0.5)
ax.set_ylabel("Best Val MSE (normalized)")
ax.set_title("Best Validation MSE")
ax.set_yscale("log")
for bar, v in zip(bars, best_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.15,
            f"{v:.6f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

# Final Train MSE
ax = axes[1]
bars = ax.bar(names, final_trains, color=bar_colors, edgecolor="black", linewidth=0.5)
ax.set_ylabel("Final Train MSE (normalized)")
ax.set_title("Final Training MSE")
ax.set_yscale("log")
for bar, v in zip(bars, final_trains):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.15,
            f"{v:.6f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

# Parameters
ax = axes[2]
bars = ax.bar(names, [p / 1e6 for p in n_params_list], color=bar_colors, edgecolor="black", linewidth=0.5)
ax.set_ylabel("Parameters (millions)")
ax.set_title("Model Size")
for bar, p in zip(bars, n_params_list):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.02,
            f"{p:,}", ha="center", va="bottom", fontsize=8)

plt.tight_layout()
plt.savefig(str(BASE / "model_comparison_bars.png"), dpi=150, bbox_inches="tight")
print("Saved: model_comparison_bars.png")
plt.close()

# ===== 6. Convergence speed =====
fig, ax = plt.subplots(figsize=(10, 6))

for name, hist in histories.items():
    val = hist["val"]
    val_norm = [v / val[0] for v in val]
    ax.plot(range(1, len(val_norm) + 1), val_norm,
            color=colors.get(name, "gray"), label=name, linewidth=2)

ax.set_xlabel("Epoch", fontsize=12)
ax.set_ylabel("Val MSE / Initial Val MSE", fontsize=12)
ax.set_title("Relative Convergence Speed", fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 1.1)

plt.tight_layout()
plt.savefig(str(BASE / "convergence_speed.png"), dpi=150, bbox_inches="tight")
print("Saved: convergence_speed.png")
plt.close()

# ===== 7. Save JSON summary =====
json_summary = {}
for name, s in summary.items():
    json_summary[name] = s

with open(BASE / "model_comparison_summary.json", "w") as f:
    json.dump(json_summary, f, indent=2)
print("\nSaved: model_comparison_summary.json")

# ===== 8. Model Analysis =====
print("\n" + "=" * 60)
print("  Model Analysis")
print("=" * 60)

# Ranking
ranked = sorted(summary.items(), key=lambda x: x[1]["best_val_mse"])
print("\n  Performance Ranking (Best Val MSE):")
for i, (name, s) in enumerate(ranked):
    print(f"    {i+1}. {name}: {s['best_val_mse']:.6f}")

print(f"\n  Recommendation:")
print(f"    - High accuracy design optimization: {ranked[0][0]}")
print(f"    - Large-scale DOE screening:         FNO (fast batch inference)")
print(f"    - Transient analysis acceleration:    Seq2SeqRNN")
print(f"    - Geometry generalization:            GINO (needs more tuning)")

print("\n" + "=" * 60)
print("  DONE - All results saved to /workspace/host_data/")
print("=" * 60)
