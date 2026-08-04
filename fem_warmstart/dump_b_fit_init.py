"""Warm-start v2: turn the best *B-predicting* surrogate into a Newton initial guess.

The section-23 benchmark seeded Newton with the curl-A checkpoint (18.7% |B|,
the section-19 negative) because it was the only model that outputs nodal A.
That conflated two questions: "is surrogate warm-starting useful?" and "is THIS
surrogate accurate enough?". The campaign's best model (doe240_bw30_ep50,
11.638% |B|) predicts element B, not A -- but `phase1_static.discrete_curl.
fit_nodal_a` inverts that: the least-squares nodal A whose P1 curl reproduces a
given element field. That is exactly the object a Newton solve wants.

Pipeline (all CPU, run inside the physicsnemo container):

    predicted element B  --lsqr-->  nodal A  --gauge anchor-->  step_<k> in the npz

Gauge: lsqr returns A up to an additive constant (minimum-norm pick). The solver
pins A = 0 on the outer stator arc, and `newton_solve` projects the guess onto
the constraint manifold, so an unanchored constant would turn into a spurious
boundary layer. Anchor by subtracting the mean of the fit over the outer-arc
nodes before saving.

The output npz is byte-compatible with `run_sweep --ai-init`, so the benchmark
protocol (identical tolerance/damping, only the guess changes) is reused as is.

    docker run --rm -e CUDA_VISIBLE_DEVICES=-1 -v "D:\\KDH\\NvidiaNemo:/workspace/app" \
      -w /workspace/app nvcr.io/nvidia/physicsnemo/physicsnemo:26.03 bash -lc \
      "python -m fem_warmstart.dump_b_fit_init --case 4 --out results/ai_init_b2a_case4.npz"
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", type=Path, default=Path("backup/doe_data"))
    ap.add_argument("--case", type=int, default=4)
    ap.add_argument("--steps", type=int, default=45)
    ap.add_argument("--ckpt", type=Path, default=Path("results/mgn_nodeB_doe240_bw30_ep50.pt"))
    ap.add_argument("--out", type=Path, default=Path("results/ai_init_b2a.npz"))
    args = ap.parse_args()

    from eval.doe_dataset import load_doe_cases
    from eval.mesh_regions import sliding_band_mask
    from eval.predictors import CurlMeshGraphNetPredictor
    from phase1_static.discrete_curl import (
        build_p1_curl_operator,
        fit_nodal_a,
        mesh_validity_mask,
    )

    manifest = json.loads((args.data_dir / "doe_manifest.json").read_text(encoding="utf-8"))
    record = load_doe_cases(manifest, args.data_dir, case_indices=[args.case]).records[0]
    mesh = record.mesh

    pred = CurlMeshGraphNetPredictor.from_checkpoint(args.ckpt, device="cpu")
    if getattr(pred, "predicts_a", False):
        raise SystemExit(f"{args.ckpt} predicts A directly; use fem_warmstart.dump_ai_init instead")

    band = sliding_band_mask(mesh.reg_code, mesh.name_of_code, mesh.moving_reg_codes)

    out = {}
    residuals = []
    n_steps = min(args.steps, len(record.samples))
    for step in range(n_steps):
        sample = record.samples[step]
        t0 = time.perf_counter()
        b_pred = pred.predict(record, sample)                     # (n_elements, 2)
        if not np.all(np.isfinite(b_pred)):
            print(f"step {step}: non-finite prediction, skipped")
            continue
        xm = np.asarray(sample.node_x_mm, dtype=np.float64) * 1e-3
        ym = np.asarray(sample.node_y_mm, dtype=np.float64) * 1e-3
        valid = mesh_validity_mask(xm, ym, mesh.tri, exclude=band)
        op = build_p1_curl_operator(xm, ym, mesh.tri, valid)
        a_fit, rel = fit_nodal_a(op, b_pred[op.element_index])
        # gauge anchor: A = 0 on the outer stator arc, matching the solver's Dirichlet
        r_mm = np.hypot(sample.node_x_mm, sample.node_y_mm)
        outer = r_mm > float(r_mm.max()) - 0.2
        a_fit = a_fit - float(a_fit[outer].mean())
        out[f"step_{step}"] = a_fit.astype(np.float64)
        residuals.append(rel)
        print(f"step {step:2d}  lsqr rel-resid {rel:.4f}  |A| max {np.abs(a_fit).max():.4e}  "
              f"{time.perf_counter()-t0:.1f}s", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **out)
    print(f"wrote {args.out} with {len(out)} steps; "
          f"mean lsqr residual {np.mean(residuals):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
