#!/usr/bin/env python3
"""
FNO (Fourier Neural Operator) training for DOE Motor FEA data.

FEM 비정형 메시를 정규 2D 격자로 보간(interpolation)한 뒤,
PhysicsNeMo FNO로 자속밀도 (Bx, By) 분포를 학습합니다.

입력: (B, 9, H, W)  - [A, J, region, time, rotate, RB, RSD, Ipk, Ph]
출력: (B, 2, H, W)  - [Bx, By]

Usage (inside Docker):
    python /workspace/train_doe_fno.py \
        --data-dir /workspace/doe_data \
        --grid-res 64 --epochs 60 --batch-size 8 --lr 1e-3

Dependencies: torch, physicsnemo, h5py, numpy, scipy
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# Shared data utilities (same directory)
sys.path.insert(0, str(Path(__file__).parent))
from doe_data_utils import load_doe_data, mesh_to_grid


def main():
    parser = argparse.ArgumentParser(description="Train FNO on DOE motor FEA data")
    parser.add_argument("--data-dir", type=str, default="/workspace/doe_data")
    parser.add_argument("--grid-res", type=int, default=64,
                        help="Regular grid resolution (H=W)")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--fno-modes", type=int, default=16,
                        help="Number of Fourier modes to keep")
    parser.add_argument("--fno-layers", type=int, default=4)
    parser.add_argument("--fno-hidden", type=int, default=64,
                        help="Latent channels in FNO")
    parser.add_argument("--max-steps-per-case", type=int, default=None)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ckpt", type=str, default="/workspace/doe_fno_ckpt.pt")
    args = parser.parse_args()

    # ---- 1. Load & convert data to regular grids ----
    records, conditions = load_doe_data(args.data_dir, args.max_steps_per_case)

    if len(records) < 2:
        print("ERROR: Need at least 2 records.")
        sys.exit(1)

    print(f"\nInterpolating {len(records)} timesteps to {args.grid_res}x{args.grid_res} grid...")
    inputs_list, targets_list = [], []
    t_interp = time.time()

    for i, (rec, cond) in enumerate(zip(records, conditions)):
        sample = mesh_to_grid(rec, cond, grid_res=args.grid_res)
        if sample is not None:
            inputs_list.append(sample["input"])    # (9, H, W)
            targets_list.append(sample["target"])  # (2, H, W)
        if (i + 1) % 100 == 0:
            print(f"  interpolated {i+1}/{len(records)}")

    print(f"  Done: {len(inputs_list)} valid grids in {time.time()-t_interp:.1f}s")

    if len(inputs_list) < 2:
        print("ERROR: Too few valid grids.")
        sys.exit(1)

    # Stack into tensors
    X = torch.from_numpy(np.stack(inputs_list, axis=0))   # (N, 9, H, W)
    Y = torch.from_numpy(np.stack(targets_list, axis=0))  # (N, 2, H, W)

    # Sanitize
    X = torch.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    Y = torch.nan_to_num(Y, nan=0.0, posinf=0.0, neginf=0.0)

    # ---- 2. Train / Val split ----
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    n_total = X.shape[0]
    indices = np.random.permutation(n_total)
    n_train = max(1, int(n_total * args.train_ratio))

    X_train, X_val = X[indices[:n_train]], X[indices[n_train:]]
    Y_train, Y_val = Y[indices[:n_train]], Y[indices[n_train:]]

    # ---- 3. Global normalization (train stats) ----
    # Per-channel mean/std over (N, H, W) for inputs
    x_mean = X_train.mean(dim=(0, 2, 3), keepdim=True)  # (1, C, 1, 1)
    x_std = X_train.std(dim=(0, 2, 3), keepdim=True).clamp_min(1e-6)
    y_mean = Y_train.mean(dim=(0, 2, 3), keepdim=True)
    y_std = Y_train.std(dim=(0, 2, 3), keepdim=True).clamp_min(1e-6)

    X_train = (X_train - x_mean) / x_std
    X_val = (X_val - x_mean) / x_std
    Y_train = (Y_train - y_mean) / y_std
    Y_val = (Y_val - y_mean) / y_std

    train_loader = DataLoader(
        TensorDataset(X_train, Y_train), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(
        TensorDataset(X_val, Y_val), batch_size=args.batch_size, shuffle=False)

    print(f"Train: {len(X_train)}, Val: {len(X_val)}")
    print(f"Input shape:  {X_train.shape}")
    print(f"Target shape: {Y_train.shape}")

    # ---- 4. Model ----
    from physicsnemo.models.fno import FNO

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = FNO(
        in_channels=X_train.shape[1],    # 9
        out_channels=Y_train.shape[1],   # 2
        dimension=2,
        latent_channels=args.fno_hidden,
        num_fno_layers=args.fno_layers,
        num_fno_modes=[args.fno_modes, args.fno_modes],
        padding=8,
        activation_fn="gelu",
        coord_features=True,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"FNO params: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)

    # ---- 5. Training loop ----
    def run_epoch(loader, training=True):
        model.train() if training else model.eval()
        total_loss, n_batch = 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            with torch.set_grad_enabled(training):
                pred = model(xb)  # (B, 2, H, W)
                loss = F.mse_loss(pred, yb)
                if torch.isnan(loss) or torch.isinf(loss):
                    continue
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
        va = run_epoch(val_loader, training=False)
        scheduler.step()
        train_hist.append(tr)
        val_hist.append(va)

        if va < best_val:
            best_val = va
            torch.save({
                "epoch": ep,
                "model_state_dict": model.state_dict(),
                "x_mean": x_mean, "x_std": x_std,
                "y_mean": y_mean, "y_std": y_std,
                "train_hist": train_hist, "val_hist": val_hist,
                "args": vars(args),
                "model_name": "FNO",
            }, args.ckpt)

        if ep == 1 or ep % 5 == 0 or ep == args.epochs:
            elapsed = time.time() - t0
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"  ep {ep:04d}/{args.epochs} | train {tr:.6f} | val {va:.6f} "
                  f"| best_val {best_val:.6f} | lr {lr_now:.2e} | {elapsed:.0f}s")

    total_time = time.time() - t0
    print(f"\n[FNO] Training complete: {args.epochs} epochs in {total_time:.1f}s")
    print(f"Best val MSE (normalized): {best_val:.6f}")
    print(f"Checkpoint: {args.ckpt}")


if __name__ == "__main__":
    main()
