"""Dump the curl-A surrogate's nodal A predictions as Newton initial guesses.

Runs the checkpoint the R4(c) experiment produced (`--target A`, section 19 of the
methodology review). That run was a negative result as a *field* surrogate --
18.7% |B| against the 12.7% target-B baseline -- but its output is a nodal A
field, which is exactly the object a Newton solve needs as u0. This is the
"second life for the failed model" the PoC design is about.

Two conventions have to be undone before the vector means anything to the solver:

* the checkpoint was trained with sector symmetry on, so its target is
  ``sign * exported A`` where ``sign`` comes from folding the rotor back into one
  sector; the export frame the solver works in needs that factor removed;
* the model output is normalised by ``a_scale``.

Run in the physicsnemo container (CPU is enough, and leaves the GPU alone):

    docker run --rm -v "D:\\KDH\\NvidiaNemo:/workspace/app" -w /workspace/app \
      -e CUDA_VISIBLE_DEVICES=-1 nvcr.io/nvidia/physicsnemo/physicsnemo:26.03 \
      bash -lc "python -m fem_warmstart.dump_ai_init --case 4 --out results/ai_init_case4.npz"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", type=Path, default=Path("backup/doe_data"))
    ap.add_argument("--case", type=int, default=4)
    ap.add_argument("--steps", type=int, default=45)
    ap.add_argument("--ckpt", type=Path, default=Path("results/mgn_nodeB_r4_curl_spectral.pt"))
    ap.add_argument("--out", type=Path, default=Path("results/ai_init.npz"))
    args = ap.parse_args()

    import torch
    from torch_geometric.data import Data

    from eval.doe_dataset import load_doe_cases
    from eval.mesh_regions import sliding_band_mask
    from eval.predictors import CurlMeshGraphNetPredictor
    from phase1_static.sector_symmetry import (
        cumulative_rotor_angle,
        rigid_rotor_node_mask,
        wrap_rotor_angle,
    )
    from train_doe_curl_mgn import build_curl_graph

    manifest = json.loads((args.data_dir / "doe_manifest.json").read_text(encoding="utf-8"))
    record = load_doe_cases(manifest, args.data_dir, case_indices=[args.case]).records[0]
    mesh = record.mesh

    pred = CurlMeshGraphNetPredictor.from_checkpoint(args.ckpt, device="cpu")
    if not getattr(pred, "predicts_a", False):
        raise SystemExit(f"{args.ckpt} does not predict nodal A; this dump needs a --target A model")

    symmetry = pred.sector_symmetry or {}
    sector_deg = float(symmetry.get("sector_deg", 45.0))
    band = sliding_band_mask(mesh.reg_code, mesh.name_of_code, mesh.moving_reg_codes)
    rigid = rigid_rotor_node_mask(mesh.tri, mesh.reg_code, mesh.name_of_code,
                                  mesh.moving_reg_codes, mesh.n_nodes)
    cum = cumulative_rotor_angle([s.rotate_step for s in record.samples])

    out = {}
    n_steps = min(args.steps, len(record.samples))
    for step in range(n_steps):
        sample = record.samples[step]
        graph = build_curl_graph(
            record, sample, band, rigid,
            cumulative_deg=float(cum[step]), sector_deg=sector_deg,
            wrap_rotor=bool(symmetry.get("wrap_rotor", True)),
            anti_periodic=bool(symmetry.get("anti_periodic_edges", True)),
            legacy_features=False,
            node_feature_names=pred.node_features or None,
            edge_feature_names=pred.edge_features or None,
        )
        if graph is None:
            print(f"step {step}: no graph, skipped")
            continue
        with torch.no_grad():
            batch = Data(
                x=torch.nan_to_num((graph.x - pred.x_mean) / pred.x_std),
                edge_index=graph.edge_index,
                edge_attr=torch.nan_to_num((graph.edge_attr - pred.e_mean) / pred.e_std),
            )
            raw = pred.model(batch.x, batch.edge_attr, batch) * pred.a_scale
        a_wrapped = raw.view(-1).double().cpu().numpy()
        _, _, sign = wrap_rotor_angle(float(cum[step]), sector_deg)
        a_export = float(sign) * a_wrapped
        out[f"step_{step}"] = a_export.astype(np.float64)
        print(f"step {step:2d}  sign {sign:+.0f}  |A| mean {np.abs(a_export).mean():.4e} "
              f"max {np.abs(a_export).max():.4e}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **out)
    print(f"wrote {args.out} with {len(out)} steps")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
