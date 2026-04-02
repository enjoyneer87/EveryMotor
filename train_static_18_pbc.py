#!/usr/bin/env python3
"""
Phase 1 Training Script: Static 1/8 Motor Model with PBC + Physics-Informed Loss.

Uses:
    - AntiPeriodicMGN  (custom_mgn.py)    : GNN with anti-periodic message passing
    - PhysicsInformedLoss (loss.py)       : data_loss + λ·physics_loss (curl A = B)
    - StaticMotorDataset (patched_datasets.py): 1/8 mesh graphs with PBC edges
    - build_graph / parse_h5_timeseries (doe_data_utils.py)

Target format in Data.y:  [Bx, By, A, J]
    - y[:, 0:1] = Bx   (magnetic flux density x-component)
    - y[:, 1:2] = By   (magnetic flux density y-component)
    - y[:, 2:3] = A    (magnetic vector potential — used for physics loss)
    - y[:, 3:4] = J    (current density)

Edge attr format (4-dim with PBC):  [dx, dy, dist, pbc_flag]

Physics loss workflow per batch:
    1.  coords = batch.pos.detach().requires_grad_(True)    # leaf tensor [N,2]
    2.  x_in   = torch.cat([coords, batch.x[:, 2:]], dim=1) # re-inject grad coords
    3.  pred   = model(x_in, batch.edge_attr, batch)         # [N, output_dim]
    4.  pred_A = pred[:, 2:3]                                # A channel
    5.  PhysicsInformedLoss computes dA/dx, dA/dy via autograd → Bx_pred, By_pred
    6.  total_loss = MSE(A_pred, A_true) + λ * (MSE(Bx_pred,Bx_true) + MSE(By_pred,By_true))

Usage:
    # Full training:
    python train_static_18_pbc.py --data-dir /workspace/doe_data --epochs 100

    # Single-batch overfit test (sanity check):
    python train_static_18_pbc.py --data-dir /workspace/doe_data --overfit-single

    # Disable physics loss (data loss only):
    python train_static_18_pbc.py --data-dir /workspace/doe_data --lambda-max 0.0
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from custom_mgn import AntiPeriodicMGN
from doe_data_utils import build_graph, parse_h5_timeseries
from loss import PhysicsInformedLoss, lambda_annealing
from patched_datasets import StaticMotorDataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sanitize(t: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)


def _compute_norm_stats(graphs, device):
    """Compute per-channel mean/std for node features, targets, and edge attrs."""
    x_cat = torch.cat([g.x for g in graphs], dim=0)
    y_cat = torch.cat([g.y for g in graphs], dim=0)
    e_cat = torch.cat([g.edge_attr for g in graphs], dim=0)

    x_mean = x_cat.mean(0, keepdim=True).to(device)
    x_std  = x_cat.std(0, keepdim=True).clamp_min(1e-6).to(device)
    y_mean = y_cat.mean(0, keepdim=True).to(device)
    y_std  = y_cat.std(0, keepdim=True).clamp_min(1e-6).to(device)
    e_mean = e_cat.mean(0, keepdim=True).to(device)
    e_std  = e_cat.std(0, keepdim=True).clamp_min(1e-6).to(device)
    return x_mean, x_std, y_mean, y_std, e_mean, e_std


def _apply_norm(graphs, x_mean, x_std, y_mean, y_std, e_mean, e_std):
    for g in graphs:
        g.x         = (g.x         - x_mean.cpu()) / x_std.cpu()
        g.y         = (g.y         - y_mean.cpu()) / y_std.cpu()
        g.edge_attr = (g.edge_attr - e_mean.cpu()) / e_std.cpu()


# ---------------------------------------------------------------------------
# Load and build graphs
# ---------------------------------------------------------------------------

def load_graphs(data_dir: Path, max_steps_per_case: Optional[int]) -> List:
    manifest_path = data_dir / "doe_manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)

    print(f"Manifest: {manifest['n_cases']} cases, "
          f"{len(manifest.get('failed', []))} failed")

    all_graphs = []
    for case in manifest["cases"]:
        h5_paths = case.get("h5_paths") or []
        if not h5_paths:
            continue

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
            h5_file = next((c for c in candidates if c.exists()), None)
            if h5_file is None:
                print(f"  [WARN] H5 not found: {h5_basename}")
                continue

            try:
                records = parse_h5_timeseries(h5_file, max_steps=max_steps_per_case)
            except Exception as exc:
                print(f"  [WARN] parse error {h5_file}: {exc}")
                continue

            for rec in records:
                g = build_graph(rec, condition, use_pbc=True)
                if g is not None:
                    g = g  # sanitize below
                    all_graphs.append(g)

        if all_graphs:
            n = case["index"]
            print(f"  case {n:04d}: total {len(all_graphs)} graphs so far | "
                  f"RB={condition.get('Ratio_Bore', '?'):.4f}")

    print(f"\nTotal graphs loaded: {len(all_graphs)}")
    return all_graphs


# ---------------------------------------------------------------------------
# Training epoch
# ---------------------------------------------------------------------------

def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer,
    criterion: PhysicsInformedLoss,
    device: torch.device,
    y_std: torch.Tensor,
    lam: float,
    training: bool = True,
) -> tuple[float, float, float]:
    """Run one training or validation epoch.

    Returns (avg_total_loss, avg_data_loss, avg_phys_loss).
    """
    model.train() if training else model.eval()
    total_sum = data_sum = phys_sum = 0.0
    n_batch = 0

    for batch in loader:
        batch = batch.to(device)

        # Re-inject coordinates as leaf tensors with requires_grad
        # so autograd can compute dA/dx, dA/dy through the model
        coords = batch.pos.detach().requires_grad_(training)  # [N, 2]
        x_in = torch.cat([coords, batch.x[:, 2:]], dim=1)    # rebuild with grad coords

        with torch.set_grad_enabled(training):
            pred = model(x_in, batch.edge_attr, batch)        # [N, output_dim]

            # y format: [Bx, By, A, J] — indices match build_graph output
            pred_A  = pred[:, 2:3]
            true_A  = batch.y[:, 2:3]
            true_Bx = batch.y[:, 0:1]
            true_By = batch.y[:, 1:2]

            total_loss, data_loss, phys_loss = criterion(
                pred_A=pred_A,
                true_A=true_A,
                coords=coords,
                true_Bx=true_Bx,
                true_By=true_By,
                lambda_val=lam,
            )

            if torch.isnan(total_loss) or torch.isinf(total_loss):
                continue

            if training:
                optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        total_sum += total_loss.item()
        data_sum  += data_loss.item()
        phys_sum  += phys_loss.item()
        n_batch   += 1

    denom = max(n_batch, 1)
    return total_sum / denom, data_sum / denom, phys_sum / denom


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Phase 1: Static 1/8 motor MeshGraphNet with PBC + physics loss"
    )
    parser.add_argument("--data-dir",          type=str,   default="/workspace/doe_data")
    parser.add_argument("--epochs",            type=int,   default=100)
    parser.add_argument("--batch-size",        type=int,   default=4)
    parser.add_argument("--lr",                type=float, default=1e-3)
    parser.add_argument("--weight-decay",      type=float, default=1e-6)
    parser.add_argument("--hidden-dim",        type=int,   default=128)
    parser.add_argument("--processor-size",    type=int,   default=15)
    parser.add_argument("--lambda-max",        type=float, default=1.0,
                        help="Maximum physics loss weight (annealed from 0)")
    parser.add_argument("--lambda-warmup",     type=int,   default=20,
                        help="Epochs to ramp lambda from 0 to lambda-max")
    parser.add_argument("--train-ratio",       type=float, default=0.8)
    parser.add_argument("--seed",              type=int,   default=42)
    parser.add_argument("--max-steps-per-case",type=int,   default=None)
    parser.add_argument("--ckpt",              type=str,
                        default="/workspace/static_18_pbc_ckpt.pt")
    parser.add_argument("--overfit-single",    action="store_true",
                        help="Overfit on a single graph (sanity check mode)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- 1. Load graphs ----
    all_graphs = load_graphs(data_dir, args.max_steps_per_case)
    if len(all_graphs) == 0:
        print("ERROR: No graphs loaded. Check data-dir and manifest.")
        sys.exit(1)

    for g in all_graphs:
        g.x         = _sanitize(g.x)
        g.y         = _sanitize(g.y)
        g.edge_attr = _sanitize(g.edge_attr)

    # ---- 2. Split ----
    if args.overfit_single:
        train_graphs = [all_graphs[0]]
        val_graphs   = [all_graphs[0]]
        print("[OVERFIT-SINGLE] Using 1 graph for both train and val.")
    else:
        indices = np.random.permutation(len(all_graphs))
        n_train = max(1, int(len(all_graphs) * args.train_ratio))
        train_graphs = [all_graphs[i] for i in indices[:n_train]]
        val_graphs   = [all_graphs[i] for i in indices[n_train:]] or [all_graphs[-1]]

    # ---- 3. Normalisation (fit on train set) ----
    x_mean, x_std, y_mean, y_std, e_mean, e_std = _compute_norm_stats(
        train_graphs, device
    )
    _apply_norm(train_graphs, x_mean, x_std, y_mean, y_std, e_mean, e_std)
    if not args.overfit_single:
        _apply_norm(val_graphs, x_mean, x_std, y_mean, y_std, e_mean, e_std)

    train_loader = DataLoader(train_graphs, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_graphs,   batch_size=args.batch_size, shuffle=False)

    sample_g = train_graphs[0]
    node_feat_dim = sample_g.x.shape[1]
    edge_feat_dim = sample_g.edge_attr.shape[1]   # 4 (includes pbc_flag)
    output_dim    = sample_g.y.shape[1]            # 4: Bx, By, A, J
    print(f"Train: {len(train_graphs)}  Val: {len(val_graphs)}")
    print(f"Node feat: {node_feat_dim}, Edge feat: {edge_feat_dim}, Output: {output_dim}")

    # ---- 4. Model ----
    model = AntiPeriodicMGN(
        input_dim_nodes=node_feat_dim,
        input_dim_edges=edge_feat_dim,   # last dim is the pbc_flag (sign)
        output_dim=output_dim,
        hidden_dim=args.hidden_dim,
        processor_size=args.processor_size,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params: {n_params:,}")

    # ---- 5. Optimiser + scheduler ----
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    criterion = PhysicsInformedLoss(lambda_physics=args.lambda_max)

    # ---- 6. Training loop ----
    best_val = float("inf")
    train_hist, val_hist = [], []
    t0 = time.time()

    for ep in range(1, args.epochs + 1):
        lam = lambda_annealing(ep, args.lambda_warmup, args.lambda_max)

        tr_total, tr_data, tr_phys = run_epoch(
            model, train_loader, optimizer, criterion, device, y_std, lam,
            training=True,
        )
        va_total, va_data, va_phys = run_epoch(
            model, val_loader, optimizer, criterion, device, y_std, lam,
            training=False,
        )
        scheduler.step()

        train_hist.append(tr_total)
        val_hist.append(va_total)

        if va_total < best_val:
            best_val = va_total
            torch.save({
                "epoch":             ep,
                "model_state_dict":  model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "x_mean": x_mean.cpu(), "x_std": x_std.cpu(),
                "y_mean": y_mean.cpu(), "y_std": y_std.cpu(),
                "e_mean": e_mean.cpu(), "e_std": e_std.cpu(),
                "train_hist": train_hist, "val_hist": val_hist,
                "args": vars(args),
                "lambda": lam,
            }, args.ckpt)

        # Print progress
        if ep == 1 or ep % 5 == 0 or ep == args.epochs:
            elapsed = time.time() - t0
            lr_now  = optimizer.param_groups[0]["lr"]
            print(
                f"  ep {ep:04d}/{args.epochs} | "
                f"train {tr_total:.4e} (data {tr_data:.4e} + phy {tr_phys:.4e}) | "
                f"val {va_total:.4e} | best {best_val:.4e} | "
                f"λ={lam:.3f} | lr {lr_now:.2e} | {elapsed:.0f}s"
            )

        # Overfit-single convergence check
        if args.overfit_single and tr_total < 1e-4:
            print(f"\n[OVERFIT-SINGLE] Converged at epoch {ep}: loss={tr_total:.2e}")
            break

    total_time = time.time() - t0
    print(f"\nTraining complete: {args.epochs} epochs in {total_time:.1f}s")
    print(f"Best val loss: {best_val:.6f}")
    print(f"Checkpoint saved: {args.ckpt}")


if __name__ == "__main__":
    main()
