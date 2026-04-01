#!/usr/bin/env python3
"""
Seq2SeqRNN training for DOE Motor FEA transient data.

PhysicsNeMo의 Seq2SeqRNN (ConvGRU 기반)을 사용하여 시변 자속밀도를 학습합니다.
FEM 메시를 정규 격자로 보간 후, 연속 타임스텝 시퀀스로 학습합니다.

입력:  (B, C_in, T_in, H, W)   이전 T_in 스텝의 [Bx, By, A, J, region]
출력:  (B, C_out, T_out, H, W)  이후 T_out 스텝의 [Bx, By]

Usage (inside Docker):
    python /workspace/train_doe_rnn.py \
        --data-dir /workspace/doe_data \
        --grid-res 64 --seq-in 4 --seq-out 4 --epochs 60

Dependencies: torch, physicsnemo, h5py, numpy, scipy
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent))
from doe_data_utils import load_doe_data, mesh_to_grid


# ---------------------------------------------------------------------------
# Temporal sequence dataset
# ---------------------------------------------------------------------------

class MotorSeqDataset(Dataset):
    """Dataset of temporal sequences from motor FEA grid data.

    Each sample: consecutive timesteps from the same DOE case,
    split into input window (T_in) and output window (T_out).

    Input channels:   [Bx, By, A, J, region] = 5  (from previous steps)
    Output channels:  [Bx, By] = 2  (future steps to predict)
    """

    def __init__(self, sequences: List[Tuple[np.ndarray, np.ndarray]]):
        """
        sequences: list of (input, target) tuples
            input:  (C_in, T_in, H, W)
            target: (C_out, T_out, H, W)
        """
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        inp, tgt = self.sequences[idx]
        return torch.from_numpy(inp), torch.from_numpy(tgt)


def build_sequences(case_grids: List[dict], seq_in: int, seq_out: int
                    ) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Build temporal sequences from a list of consecutive grid snapshots.

    Input channels per step: [Bx, By, A, J, region] = first 5 channels
    of the grid input + the 2 target channels.
    """
    if len(case_grids) < seq_in + seq_out:
        return []

    sequences = []
    for start in range(len(case_grids) - seq_in - seq_out + 1):
        # Collect input and target windows
        inp_frames = []
        tgt_frames = []

        for t in range(start, start + seq_in):
            g = case_grids[t]
            # RNN input: previous physical state = [Bx(2), By(2), A(0), J(1), region(2)]
            # from input grid [A, J, region, time, rot, RB, RSD, Ipk, Ph] and target [Bx, By]
            bx_by = g["target"]          # (2, H, W)
            a_j_reg = g["input"][:3]     # (3, H, W): A, J, region
            frame = np.concatenate([bx_by, a_j_reg], axis=0)  # (5, H, W)
            inp_frames.append(frame)

        for t in range(start + seq_in, start + seq_in + seq_out):
            g = case_grids[t]
            tgt_frames.append(g["target"])  # (2, H, W)

        # Stack: (C, T, H, W)
        inp = np.stack(inp_frames, axis=1)  # (5, T_in, H, W)
        tgt = np.stack(tgt_frames, axis=1)  # (2, T_out, H, W)
        sequences.append((inp.astype(np.float32), tgt.astype(np.float32)))

    return sequences


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train Seq2SeqRNN on DOE motor FEA transient data")
    parser.add_argument("--data-dir", type=str, default="/workspace/doe_data")
    parser.add_argument("--grid-res", type=int, default=64)
    parser.add_argument("--seq-in", type=int, default=4,
                        help="Number of input timesteps")
    parser.add_argument("--seq-out", type=int, default=4,
                        help="Number of output timesteps to predict")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--hidden-channels", type=int, default=64,
                        help="Latent channels in ConvGRU")
    parser.add_argument("--max-steps-per-case", type=int, default=None)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ckpt", type=str, default="/workspace/doe_rnn_ckpt.pt")
    args = parser.parse_args()

    import json
    data_dir = Path(args.data_dir)
    manifest_path = data_dir / "doe_manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)

    print(f"Manifest: {manifest['n_cases']} cases")

    # ---- 1. Load data case-by-case (preserving temporal order) ----
    all_case_sequences = []
    t0_load = time.time()

    for case in manifest["cases"]:
        h5_paths = case.get("h5_paths") or []
        if not h5_paths:
            continue

        condition = {}
        condition.update(case.get("geometry", {}))
        condition.update(case.get("electrical", {}))

        case_grids = []
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

            from doe_data_utils import parse_h5_timeseries
            try:
                records = parse_h5_timeseries(h5_file, max_steps=args.max_steps_per_case)
            except Exception as e:
                print(f"  [WARN] {h5_file}: {e}")
                continue

            for rec in records:
                grid = mesh_to_grid(rec, condition, grid_res=args.grid_res)
                if grid is not None:
                    case_grids.append(grid)

        if len(case_grids) >= args.seq_in + args.seq_out:
            seqs = build_sequences(case_grids, args.seq_in, args.seq_out)
            all_case_sequences.extend(seqs)
            print(f"  case {case['index']:04d}: {len(case_grids)} grids → {len(seqs)} sequences")

    print(f"\nTotal sequences: {len(all_case_sequences)} "
          f"(loaded in {time.time()-t0_load:.1f}s)")

    if len(all_case_sequences) < 2:
        print("ERROR: Need at least 2 sequences.")
        sys.exit(1)

    # ---- 2. Train / Val split ----
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    n_total = len(all_case_sequences)
    indices = np.random.permutation(n_total)
    n_train = max(1, int(n_total * args.train_ratio))

    train_seqs = [all_case_sequences[i] for i in indices[:n_train]]
    val_seqs = [all_case_sequences[i] for i in indices[n_train:]]

    # ---- 3. Normalization ----
    # Per-channel stats over all training input/target frames
    all_inp = np.concatenate([s[0] for s in train_seqs], axis=1)  # (5, T_total, H, W)
    all_tgt = np.concatenate([s[1] for s in train_seqs], axis=1)  # (2, T_total, H, W)

    inp_mean = all_inp.mean(axis=(1, 2, 3), keepdims=True).astype(np.float32)  # (5,1,1,1)
    inp_std = np.clip(all_inp.std(axis=(1, 2, 3), keepdims=True), 1e-6, None).astype(np.float32)
    tgt_mean = all_tgt.mean(axis=(1, 2, 3), keepdims=True).astype(np.float32)  # (2,1,1,1)
    tgt_std = np.clip(all_tgt.std(axis=(1, 2, 3), keepdims=True), 1e-6, None).astype(np.float32)

    del all_inp, all_tgt

    def norm_seqs(seqs):
        return [((inp - inp_mean) / inp_std, (tgt - tgt_mean) / tgt_std)
                for inp, tgt in seqs]

    train_seqs = norm_seqs(train_seqs)
    val_seqs = norm_seqs(val_seqs)

    train_loader = DataLoader(
        MotorSeqDataset(train_seqs), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(
        MotorSeqDataset(val_seqs), batch_size=args.batch_size, shuffle=False)

    c_in = train_seqs[0][0].shape[0]   # 5
    c_out = train_seqs[0][1].shape[0]  # 2
    print(f"Train: {len(train_seqs)}, Val: {len(val_seqs)}")
    print(f"Input: ({c_in}, {args.seq_in}, {args.grid_res}, {args.grid_res})")
    print(f"Target: ({c_out}, {args.seq_out}, {args.grid_res}, {args.grid_res})")

    # ---- 4. Model ----
    from physicsnemo.models.rnn import Seq2SeqRNN

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = Seq2SeqRNN(
        input_channels=c_in,          # 5: [Bx, By, A, J, region]
        dimension=2,
        nr_latent_channels=args.hidden_channels,
        nr_tsteps=args.seq_out,        # predict this many future steps
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Seq2SeqRNN params: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)

    # ---- 5. Training loop ----
    def run_epoch(loader, training=True):
        model.train() if training else model.eval()
        total_loss, n_batch = 0.0, 0

        for xb, yb in loader:
            xb = xb.to(device)  # (B, C_in, T_in, H, W)
            yb = yb.to(device)  # (B, C_out, T_out, H, W)

            with torch.set_grad_enabled(training):
                # Seq2SeqRNN expects (N, C, T, H, W) and outputs (N, C, T_out, H, W)
                pred = model(xb)

                # Output has input_channels, slice to match target channels
                pred_bxby = pred[:, :c_out, :, :, :]  # (B, 2, T_out, H, W)
                loss = F.mse_loss(pred_bxby, yb)

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
                "inp_mean": inp_mean, "inp_std": inp_std,
                "tgt_mean": tgt_mean, "tgt_std": tgt_std,
                "train_hist": train_hist, "val_hist": val_hist,
                "args": vars(args),
                "model_name": "Seq2SeqRNN",
            }, args.ckpt)

        if ep == 1 or ep % 5 == 0 or ep == args.epochs:
            elapsed = time.time() - t0
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"  ep {ep:04d}/{args.epochs} | train {tr:.6f} | val {va:.6f} "
                  f"| best_val {best_val:.6f} | lr {lr_now:.2e} | {elapsed:.0f}s")

    total_time = time.time() - t0
    print(f"\n[Seq2SeqRNN] Training complete: {args.epochs} epochs in {total_time:.1f}s")
    print(f"Best val MSE (normalized): {best_val:.6f}")
    print(f"Checkpoint: {args.ckpt}")


if __name__ == "__main__":
    main()
