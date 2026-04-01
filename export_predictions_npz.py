#!/usr/bin/env python3
r"""
Batch inference → NPZ export for interactive GUI comparison.

For each DOE case, runs MeshGraphNet prediction on every timestep and saves:
  case_XXXX_predictions.npz  containing:
    pos:       (n_steps, n_nodes, 2)   node x,y positions [mm]
    true_bxy:  (n_steps, n_nodes, 2)   FEA ground truth [Bx, By] [T]
    pred_bxy:  (n_steps, n_nodes, 2)   MeshGraphNet prediction [Bx, By] [T]
    steps:     (n_steps,)              step indices
    times:     (n_steps,)              time [s]
    condition: JSON string             {Ratio_Bore, Ratio_SlotDepth_ParallelSlot,
                                        PeakCurrent, PhaseAdvance}

Usage (inside Docker):
    python export_predictions_npz.py \
        --ckpt /workspace/doe_meshgraphnet_ckpt.pt \
        --data-dir /workspace/doe_data \
        --out-dir /workspace/pred_npz \
        [--cases 0 5 10 20]   # specific cases, or omit for ALL
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

# Reuse parser + graph builder from training script
sys.path.insert(0, str(Path(__file__).parent))
from train_doe_meshgraphnet import parse_h5_timeseries, build_graph
from infer_doe_meshgraphnet import load_model_and_stats, predict_one


def main():
    parser = argparse.ArgumentParser(description="Batch predict → NPZ for GUI comparison")
    parser.add_argument("--ckpt", type=str, default="/workspace/doe_meshgraphnet_ckpt.pt")
    parser.add_argument("--data-dir", type=str, default="/workspace/doe_data")
    parser.add_argument("--out-dir", type=str, default="/workspace/pred_npz")
    parser.add_argument("--cases", type=int, nargs="*", default=None,
                        help="Specific case indices (default=all)")
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    # Load model
    model, norm, ckpt = load_model_and_stats(args.ckpt, device)

    # Load manifest
    data_dir = Path(args.data_dir)
    with open(data_dir / "doe_manifest.json") as f:
        manifest = json.load(f)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Filter cases
    cases = manifest["cases"]
    if args.cases is not None:
        case_set = set(args.cases)
        cases = [c for c in cases if c["index"] in case_set]
    print(f"Processing {len(cases)} cases...", flush=True)

    t0 = time.time()
    for ci, case in enumerate(cases):
        idx = case["index"]
        h5_paths = case.get("h5_paths") or []
        condition = {}
        condition.update(case.get("geometry", {}))
        condition.update(case.get("electrical", {}))

        # Find H5
        h5_file = None
        for h5p in h5_paths:
            h5_basename = h5p.replace("\\", "/").split("/")[-1]
            for c in [Path(h5p),
                      data_dir / f"case_{idx:04d}" / "postproc" / h5_basename,
                      data_dir / h5_basename]:
                if c.exists():
                    h5_file = c
                    break
            if h5_file:
                break

        if h5_file is None:
            print(f"  [SKIP] case {idx}: no H5 found", flush=True)
            continue

        # Parse all timesteps
        try:
            records = parse_h5_timeseries(h5_file, max_steps=args.max_steps)
        except Exception as e:
            print(f"  [SKIP] case {idx}: parse error: {e}", flush=True)
            continue

        n_steps = len(records)
        if n_steps == 0:
            continue

        # First graph to get n_nodes
        g0 = build_graph(records[0], condition)
        if g0 is None:
            continue
        n_nodes = g0.pos.shape[0]

        pos_arr = np.zeros((n_steps, n_nodes, 2), dtype=np.float32)
        true_arr = np.zeros((n_steps, n_nodes, 2), dtype=np.float32)
        pred_arr = np.zeros((n_steps, n_nodes, 2), dtype=np.float32)
        step_keys = np.zeros(n_steps, dtype=np.int32)
        times = np.zeros(n_steps, dtype=np.float64)

        for si, rec in enumerate(records):
            g = build_graph(rec, condition) if si > 0 else g0
            if g is None:
                continue

            pred_bxy, true_bxy = predict_one(model, g, norm, device)
            pos_arr[si] = g.pos.numpy()
            true_arr[si] = true_bxy
            pred_arr[si] = pred_bxy
            step_keys[si] = rec.get("step_key", si)
            times[si] = rec.get("time_s", 0.0)

        # Save NPZ
        npz_path = out_dir / f"case_{idx:04d}_predictions.npz"
        np.savez_compressed(
            npz_path,
            pos=pos_arr,
            true_bxy=true_arr,
            pred_bxy=pred_arr,
            steps=step_keys,
            times=times,
            condition=json.dumps(condition),
            case_index=idx,
        )
        elapsed = time.time() - t0
        print(f"  case {idx:04d}: {n_steps} steps, {n_nodes} nodes → {npz_path.name} "
              f"[{elapsed:.0f}s]", flush=True)

    total = time.time() - t0
    print(f"\nDone: {len(cases)} cases in {total:.0f}s → {out_dir}", flush=True)


if __name__ == "__main__":
    main()
