#!/usr/bin/env python3
"""
GINO (Geometry-Informed Neural Operator) training for DOE Motor FEA data.

neuraloperator 라이브러리의 GINO를 사용하여 비정형 FEM 메시에서 직접
자속밀도 (Bx, By) 분포를 학습합니다.

GINO 아키텍처:
  Input GNO (비정형 메시 → 잠재 정규 격자)
  → FNO Blocks (주파수 도메인 학습)
  → Output GNO (잠재 격자 → 비정형 메시)

입력: input_geom (N, 2), features (N, 9), latent_queries (R, R, 2)
출력: (N, 2) - [Bx, By]

Usage (inside Docker):
    pip install neuraloperator
    python /workspace/train_doe_gino.py \
        --data-dir /workspace/doe_data \
        --latent-res 32 --epochs 60 --lr 1e-3

Dependencies: torch, neuraloperator, h5py, numpy, scipy
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from doe_data_utils import load_doe_data, build_gino_sample


# ---------------------------------------------------------------------------
# GINO Dataset wrapper
# ---------------------------------------------------------------------------

class MotorGINODataset(torch.utils.data.Dataset):
    """Dataset that holds pre-built GINO samples in memory."""

    def __init__(self, samples: List[dict]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "input_geom": torch.from_numpy(s["input_geom"]),        # (N, 2)
            "features": torch.from_numpy(s["features"]),            # (N, C_in)
            "target": torch.from_numpy(s["target"]),                # (N, 2)
            "latent_queries": torch.from_numpy(s["latent_queries"]),# (R, R, 2)
        }


def gino_collate_fn(batch):
    """Custom collate: GINO expects batch_size=1 with shared geometry.
    We process samples one at a time within a mini-batch."""
    return batch  # list of dicts


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def compute_stats(samples: List[dict], key: str):
    """Compute mean/std over all samples for a given key."""
    all_data = np.concatenate([s[key] for s in samples], axis=0)
    mean = all_data.mean(axis=0, keepdims=True).astype(np.float32)
    std = np.clip(all_data.std(axis=0, keepdims=True), 1e-6, None).astype(np.float32)
    return mean, std


def normalize_samples(samples, feat_mean, feat_std, tgt_mean, tgt_std):
    """In-place normalization."""
    for s in samples:
        s["features"] = ((s["features"] - feat_mean) / feat_std).astype(np.float32)
        s["target"] = ((s["target"] - tgt_mean) / tgt_std).astype(np.float32)
    return samples


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train GINO on DOE motor FEA data")
    parser.add_argument("--data-dir", type=str, default="/workspace/doe_data")
    parser.add_argument("--latent-res", type=int, default=32,
                        help="Latent grid resolution for GINO")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--fno-modes", type=int, default=16,
                        help="Fourier modes in latent FNO")
    parser.add_argument("--fno-layers", type=int, default=4)
    parser.add_argument("--fno-hidden", type=int, default=64)
    parser.add_argument("--gno-radius", type=float, default=0.1,
                        help="GNO neighbor search radius (in normalized [0,1] coords)")
    parser.add_argument("--max-steps-per-case", type=int, default=None)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ckpt", type=str, default="/workspace/doe_gino_ckpt.pt")
    args = parser.parse_args()

    # ---- 1. Load & convert data ----
    records, conditions = load_doe_data(args.data_dir, args.max_steps_per_case)
    if len(records) < 2:
        print("ERROR: Need at least 2 records.")
        sys.exit(1)

    print(f"\nBuilding GINO samples (latent_res={args.latent_res})...")
    samples = []
    t0_build = time.time()
    for i, (rec, cond) in enumerate(zip(records, conditions)):
        s = build_gino_sample(rec, cond, latent_res=args.latent_res)
        if s is not None:
            samples.append(s)
        if (i + 1) % 100 == 0:
            print(f"  built {i+1}/{len(records)}")
    print(f"  Done: {len(samples)} samples in {time.time()-t0_build:.1f}s")

    if len(samples) < 2:
        print("ERROR: Too few valid samples.")
        sys.exit(1)

    # ---- 2. Train / Val split ----
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    n_total = len(samples)
    indices = np.random.permutation(n_total)
    n_train = max(1, int(n_total * args.train_ratio))

    train_samples = [samples[i] for i in indices[:n_train]]
    val_samples = [samples[i] for i in indices[n_train:]]

    # ---- 3. Normalization ----
    feat_mean, feat_std = compute_stats(train_samples, "features")
    tgt_mean, tgt_std = compute_stats(train_samples, "target")

    normalize_samples(train_samples, feat_mean, feat_std, tgt_mean, tgt_std)
    normalize_samples(val_samples, feat_mean, feat_std, tgt_mean, tgt_std)

    train_ds = MotorGINODataset(train_samples)
    val_ds = MotorGINODataset(val_samples)

    # GINO processes one geometry at a time (batch_size=1 for geometry)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=1, shuffle=True, collate_fn=gino_collate_fn)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=1, shuffle=False, collate_fn=gino_collate_fn)

    in_channels = train_samples[0]["features"].shape[1]  # 9
    out_channels = train_samples[0]["target"].shape[1]    # 2
    print(f"Train: {len(train_samples)}, Val: {len(val_samples)}")
    print(f"In channels: {in_channels}, Out channels: {out_channels}")
    print(f"Nodes per sample: ~{train_samples[0]['input_geom'].shape[0]}")

    # ---- 4. Model ----
    from neuralop.models import GINO

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = GINO(
        in_channels=in_channels,       # 9
        out_channels=out_channels,     # 2
        gno_coord_dim=2,               # 2D motor cross-section
        in_gno_radius=args.gno_radius,
        out_gno_radius=args.gno_radius,
        fno_in_channels=in_channels,   # must match in_channels for linear GNO
        fno_n_modes=(args.fno_modes, args.fno_modes),
        fno_hidden_channels=args.fno_hidden,
        fno_n_layers=args.fno_layers,
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
        gno_use_open3d=False,          # 2D data, no Open3D needed
        gno_use_torch_scatter=True,    # use torch_scatter for fast neighbor ops
        fno_use_channel_mlp=True,
        fno_channel_mlp_expansion=0.5,
        fno_non_linearity=F.gelu,
        fno_norm=None,
        fno_skip="linear",
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"GINO params: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    # ---- 5. Training loop ----
    def run_epoch(loader, training=True):
        model.train() if training else model.eval()
        total_loss, n_sample = 0.0, 0

        for batch_list in loader:
            for sample in batch_list:
                input_geom = sample["input_geom"].unsqueeze(0).to(device)      # (1, N, 2)
                latent_q = sample["latent_queries"].unsqueeze(0).to(device)    # (1, R, R, 2)
                output_q = sample["input_geom"].unsqueeze(0).to(device)        # (1, N, 2) same as input
                x_feat = sample["features"].unsqueeze(0).to(device)            # (1, N, C_in)
                y_true = sample["target"].unsqueeze(0).to(device)              # (1, N, 2)

                with torch.set_grad_enabled(training):
                    # GINO forward: input_geom, latent_queries, output_queries, x=features
                    pred = model(
                        input_geom=input_geom,
                        latent_queries=latent_q,
                        output_queries=output_q,
                        x=x_feat,
                    )  # (1, N, out_channels)

                    loss = F.mse_loss(pred.squeeze(0), y_true.squeeze(0))

                    if torch.isnan(loss) or torch.isinf(loss):
                        continue

                    if training:
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()

                total_loss += loss.item()
                n_sample += 1

        return total_loss / max(n_sample, 1)

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
                "feat_mean": feat_mean, "feat_std": feat_std,
                "tgt_mean": tgt_mean, "tgt_std": tgt_std,
                "train_hist": train_hist, "val_hist": val_hist,
                "args": vars(args),
                "model_name": "GINO",
            }, args.ckpt)

        if ep == 1 or ep % 5 == 0 or ep == args.epochs:
            elapsed = time.time() - t0
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"  ep {ep:04d}/{args.epochs} | train {tr:.6f} | val {va:.6f} "
                  f"| best_val {best_val:.6f} | lr {lr_now:.2e} | {elapsed:.0f}s")

    total_time = time.time() - t0
    print(f"\n[GINO] Training complete: {args.epochs} epochs in {total_time:.1f}s")
    print(f"Best val MSE (normalized): {best_val:.6f}")
    print(f"Checkpoint: {args.ckpt}")


if __name__ == "__main__":
    main()
