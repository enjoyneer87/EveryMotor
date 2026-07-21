#!/usr/bin/env python3
"""Inference script for phase1_static PBC-aware MeshGraphNet (PBC-MGN).

Model name: **SymMGN** (Symmetry-aware MeshGraphNet)
  - Encodes 1/8-sector anti-periodic boundary via explicit PBC edge_attr=-1.0 edges
  - Uses PhysicsNeMo MeshGraphNet backbone
  - Hybrid loss: supervised A + supervised B + curl(A) consistency

Architecture comparison vs other models:
  MGN(original) : graph edges, no PBC topology, no anti-periodic sign
  FNO           : regular grid, PBC via periodic padding in frequency domain
  RNN           : sequential grid, same as FNO
  GINO          : GNO encoder + FNO latent, PBC via ghost-node boundary copies
  SymMGN (ours) : graph edges + PBC edges with anti-periodic sign=-1.0  ← this script

Usage (inside Docker):
  docker exec physicsnemo bash -lc "
    cd /workspace/app;
    python infer_phase1_pbc.py
      --ckpt /workspace/app/results/symm_mgn_pbc.pt
      --data /workspace/app/results/overfit_smoke.npz
      --out /workspace/app/results/symm_mgn_infer.npz
  "

For standalone DOE data:
  python infer_phase1_pbc.py \\
    --ckpt results/symm_mgn_pbc.pt \\
    --data-dir doe_data \\
    --case-idx 0 \\
    --out results/symm_mgn_infer_case0.npz
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch_geometric.loader import DataLoader

sys.path.insert(0, str(Path(__file__).parent))

from phase1_static.train import build_model, forward_model
from phase1_static.motor_dataset import (
    StaticMotorDataset,
    build_samples_from_npz,
    build_samples_from_doe_manifest,
)
from phase1_static.contracts import normalize_channels_to_bx_by_a_je

# PhysicsNeMo emits DGL backend warnings even when the PyG execution path is used.
warnings.filterwarnings(
    "ignore",
    message=r"MeshGraphNet \(DGL version\) requires the DGL library\.",
    category=UserWarning,
)


CHANNEL_ORDER = ["Bx", "By", "A", "Je"]
CHANNEL_KEYS = tuple(channel.lower() for channel in CHANNEL_ORDER)


# ─── Checkpoint I/O ─────────────────────────────────────────────────────────

def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    epoch: int,
    best_loss: float,
    args_dict: dict,
) -> None:
    """Save SymMGN checkpoint in a format compatible with load_symm_mgn."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_name": "SymMGN",
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "best_loss": best_loss,
            "args": args_dict,
            "channel_order": CHANNEL_ORDER,
            "pbc_enabled": True,
            "pbc_rotation_deg": -45.0,
        },
        path,
    )


def load_symm_mgn(
    ckpt_path: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, dict]:
    """Load SymMGN checkpoint. Raises RuntimeError if not a SymMGN checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_name = ckpt.get("model_name", "unknown")
    if model_name != "SymMGN":
        raise RuntimeError(
            f"Expected SymMGN checkpoint, got '{model_name}'. "
            "Use infer_doe_meshgraphnet.py for old-style MGN checkpoints."
        )

    saved_args = ckpt.get("args", {})
    input_dim = saved_args.get("input_dim", None)
    hidden_dim = saved_args.get("hidden_dim", 128)
    channel_order = ckpt.get("channel_order") or saved_args.get("channel_order") or CHANNEL_ORDER
    output_dim = int(saved_args.get("output_dim", len(channel_order)))

    if input_dim is None:
        raise RuntimeError(
            "Checkpoint missing 'args.input_dim'. "
            "Re-train with the current phase1_static.train to regenerate the checkpoint."
        )

    model = build_model(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model, ckpt


# ─── Inference loop ─────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    samples: list,
    device: torch.device,
    batch_size: int = 8,
) -> Dict[str, np.ndarray]:
    """Run SymMGN inference on all samples. Returns per-node predictions."""
    dataset = StaticMotorDataset(samples)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_pred = []
    all_gt = []
    all_pos = []

    for batch in loader:
        batch = batch.to(device)
        # pos must be leaf tensor (not requires_grad for inference)
        batch.pos = batch.pos.detach()
        # Re-attach pos as first columns of x for coordinate channels
        d = batch.pos.shape[1]
        batch.x = torch.cat([batch.pos, batch.x[:, d:]], dim=1)

        pred_raw = forward_model(model, batch)
        pred, _ = normalize_channels_to_bx_by_a_je(pred_raw)
        gt, _ = normalize_channels_to_bx_by_a_je(batch.y)

        all_pred.append(pred.detach().cpu().numpy())
        all_gt.append(gt.detach().cpu().numpy())
        all_pos.append(batch.pos.detach().cpu().numpy())

    pred_np = np.concatenate(all_pred, axis=0)   # [N_total, 4]
    gt_np = np.concatenate(all_gt, axis=0)        # [N_total, 4]
    pos_np = np.concatenate(all_pos, axis=0)      # [N_total, 2]

    result = {
        "pos_x": pos_np[:, 0],
        "pos_y": pos_np[:, 1],
    }
    for channel_index, channel_name in enumerate(CHANNEL_KEYS):
        result[f"gt_{channel_name}"] = gt_np[:, channel_index]
        result[f"pred_{channel_name}"] = pred_np[:, channel_index]
    return result


def compute_per_channel_metrics(result: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Compute RMSE and relative error per channel."""
    metrics = {}
    for ch in CHANNEL_KEYS:
        gt = result[f"gt_{ch}"]
        pred = result[f"pred_{ch}"]
        err = gt - pred
        rmse = float(np.sqrt(np.mean(err ** 2)))
        rel = float(rmse / (np.std(gt) + 1e-8))
        metrics[f"rmse_{ch}"] = rmse
        metrics[f"rel_err_{ch}"] = rel
    return metrics


# ─── NPZ save ───────────────────────────────────────────────────────────────

def save_infer_npz(
    out_path: Path,
    result: Dict[str, np.ndarray],
    metrics: Dict[str, float],
    meta: dict,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        pos_x=result["pos_x"],
        pos_y=result["pos_y"],
        gt_bx=result["gt_bx"], gt_by=result["gt_by"],
        gt_a=result["gt_a"], gt_je=result["gt_je"],
        pred_bx=result["pred_bx"], pred_by=result["pred_by"],
        pred_a=result["pred_a"], pred_je=result["pred_je"],
        metrics=json.dumps(metrics),
        meta=json.dumps(meta),
    )


def filter_samples_by_step_index(
    samples: list,
    step_index: int | None,
) -> list:
    """Return only samples matching the requested step index."""
    if step_index is None:
        return samples
    return [
        sample
        for sample in samples
        if int(sample.get("step_index", -1)) == int(step_index)
    ]


# ─── CLI ────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="SymMGN (PBC-aware MeshGraphNet) inference")
    ap.add_argument("--ckpt", required=True, help="Path to SymMGN checkpoint (.pt)")
    # Data source (mutually exclusive in practice; --data takes priority)
    ap.add_argument("--data", default=None, help="NPZ bundle path (phase1_static format)")
    ap.add_argument("--data-dir", default="doe_data", help="DOE root dir with doe_manifest.json")
    ap.add_argument("--case-idx", type=int, default=0, help="DOE case index (used with --data-dir)")
    ap.add_argument(
        "--source-file-types",
        nargs="+",
        default=None,
        help="Optional MotorCAD source file classes to include, e.g. OnLoadTorque StaticLoad",
    )
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument(
        "--step-index",
        type=int,
        default=None,
        help="Optional explicit step_index to select after DOE samples are loaded.",
    )
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--out", default="results/symm_mgn_infer.npz")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[SymMGN infer] device={device}")

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        print(f"[ERROR] Checkpoint not found: {ckpt_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading SymMGN checkpoint: {ckpt_path}")
    model, ckpt = load_symm_mgn(ckpt_path, device)
    print(f"  epoch={ckpt.get('epoch')}, best_loss={ckpt.get('best_loss'):.6f}")

    if args.data and Path(args.data).exists():
        print(f"Loading NPZ: {args.data}")
        samples = build_samples_from_npz(args.data)
    else:
        print(f"Loading DOE manifest: {args.data_dir} case={args.case_idx}")
        samples = build_samples_from_doe_manifest(
            args.data_dir,
            args.max_steps,
            case_indices=[args.case_idx],
            source_file_types=args.source_file_types,
        )

    samples = filter_samples_by_step_index(samples, args.step_index)
    if not samples:
        requested = (
            f" case={args.case_idx} step_index={args.step_index}"
            if args.step_index is not None else ""
        )
        print(f"[ERROR] No samples found for{requested}", file=sys.stderr)
        sys.exit(1)

    print(f"  {len(samples)} samples loaded")

    t0 = time.time()
    result = run_inference(model, samples, device, batch_size=args.batch_size)
    elapsed = time.time() - t0
    print(f"  Inference done in {elapsed:.2f}s ({len(samples) / elapsed:.1f} samples/s)")

    metrics = compute_per_channel_metrics(result)
    print("  Per-channel RMSE:")
    for ch in CHANNEL_KEYS:
        print(f"    {ch}: RMSE={metrics[f'rmse_{ch}']:.5f}  rel={metrics[f'rel_err_{ch}']:.3f}")

    meta = {
        "model_name": "SymMGN",
        "ckpt": str(ckpt_path),
        "case_idx": args.case_idx,
        "n_samples": len(samples),
        "step_idx": (
            int(samples[0].get("step_index", -1))
            if len({int(sample.get("step_index", -1)) for sample in samples}) == 1
            else None
        ),
        "step_indices": sorted({int(sample.get("step_index", -1)) for sample in samples}),
        "elapsed_s": round(elapsed, 3),
        "channel_order": ckpt.get("channel_order", CHANNEL_ORDER),
        "pbc_enabled": True,
        "pbc_rotation_deg": -45.0,
        "source_file_types": args.source_file_types or [],
    }

    out_path = Path(args.out)
    save_infer_npz(out_path, result, metrics, meta)
    print(f"  Saved: {out_path} ({out_path.stat().st_size / 1e6:.2f} MB)")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
