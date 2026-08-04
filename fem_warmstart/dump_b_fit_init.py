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


def _gap_harmonic_patch(a_fit, op, b_pred, mesh, sample, n_orders):
    """Overwrite the fit's unconstrained gap nodes with an analytical Laplace field.

    The surrogate predicts nothing on the sliding-band layers (a2/a3/a4), so the
    lsqr fit leaves their interior nodes at the minimum-norm value -- physically
    arbitrary, and the air gap's nu0 makes exactly those nodes dominate the
    Newton residual (measured: initial ||R||/||f|| ~ 134 with the plain fit).
    The gap is a source-free annulus, so A there is a harmonic series in the
    anti-periodic orders n in {4,12,20,...}. Fit the series to the *predicted*
    a1-band B (the one gap layer the model does predict), evaluate A analytically
    at the orphan nodes, and stitch it in.

    The alpha/beta (growing/decaying) split is nearly collinear across a1's
    0.25 mm -- lstsq's pseudo-inverse picks the balanced minimum-norm split,
    which reconstructs a1 correctly and extrapolates the low orders (91% of the
    energy is n=4) well; high-order extrapolation error lands in a region Newton
    repairs cheaply anyway.
    """
    tri = np.stack(mesh.tri, axis=1)
    code_of = {str(v).strip().lower(): int(k) for k, v in mesh.name_of_code.items()}
    a1 = np.asarray(mesh.reg_code) == code_of["a1"]

    x_mm = np.asarray(sample.node_x_mm, dtype=np.float64)
    y_mm = np.asarray(sample.node_y_mm, dtype=np.float64)
    cx = x_mm[tri].mean(axis=1) * 1e-3
    cy = y_mm[tri].mean(axis=1) * 1e-3
    r = np.hypot(cx, cy)[a1]
    th = np.arctan2(cy, cx)[a1]
    r0 = float(r.mean())
    orders = np.array([4 * (2 * k + 1) for k in range(n_orders)], dtype=np.float64)

    # columns: for each n, [grow*cos, grow*sin, decay*cos, decay*sin] of A;
    # B_r = (1/r) dA/dth, B_th = -dA/dr; then to cartesian.
    def basis(rr, tt):
        cols_bx, cols_by, cols_a = [], [], []
        ct, st = np.cos(tt), np.sin(tt)
        for n in orders:
            g, d = (rr / r0) ** n, (r0 / rr) ** n
            for rad, drad in ((g, n * g / rr), (d, -n * d / rr)):
                for ang, dang in ((np.cos(n * tt), -n * np.sin(n * tt)),
                                  (np.sin(n * tt), n * np.cos(n * tt))):
                    br = rad * dang / rr
                    bt = -drad * ang
                    cols_bx.append(br * ct - bt * st)
                    cols_by.append(br * st + bt * ct)
                    cols_a.append(rad * ang)
        return (np.stack(cols_bx, axis=1), np.stack(cols_by, axis=1),
                np.stack(cols_a, axis=1))

    bx_cols, by_cols, _ = basis(r, th)
    design = np.concatenate([bx_cols, by_cols], axis=0)
    rhs = np.concatenate([b_pred[a1, 0], b_pred[a1, 1]])
    coef, *_ = np.linalg.lstsq(design, rhs, rcond=1e-8)
    fit_rel = float(np.linalg.norm(design @ coef - rhs) / np.linalg.norm(rhs))

    # orphan nodes = touched only by excluded elements; all strictly inside the gap
    fitted_nodes = np.zeros(a_fit.size, dtype=bool)
    fitted_nodes[op.node_index.ravel()] = True
    orphan = ~fitted_nodes
    rn = np.hypot(x_mm, y_mm) * 1e-3
    _, _, a_cols = basis(rn[orphan], np.arctan2(y_mm, x_mm)[orphan])
    a_harm = a_cols @ coef

    # gauge continuity: compare on fitted nodes inside the gap span
    gap_lo, gap_hi = rn[orphan].min() - 3e-4, rn[orphan].max() + 3e-4
    boundary = fitted_nodes & (rn > gap_lo) & (rn < gap_hi)
    _, _, a_cols_b = basis(rn[boundary], np.arctan2(y_mm, x_mm)[boundary])
    offset = float(np.median(a_fit[boundary] - a_cols_b @ coef))

    a_out = a_fit.copy()
    a_out[orphan] = a_harm + offset
    return a_out, fit_rel, int(orphan.sum())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", type=Path, default=Path("backup/doe_data"))
    ap.add_argument("--case", type=int, default=4)
    ap.add_argument("--steps", type=int, default=45)
    ap.add_argument("--ckpt", type=Path, default=Path("results/mgn_nodeB_doe240_bw30_ep50.pt"))
    ap.add_argument("--gap-harmonic", type=int, default=0, metavar="N_ORDERS",
                    help="stitch an analytical Laplace gap field (N anti-periodic "
                         "orders fitted to the predicted a1-band B) into the "
                         "otherwise-unconstrained sliding-band nodes")
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
        # The predictor returns NaN on excluded elements (sliding band and friends)
        # BY DESIGN -- that is what the benchmark's coverage 0.865 is. Fit only on
        # elements that are both geometrically valid and actually predicted.
        finite = np.isfinite(b_pred).all(axis=1)
        if not finite.any():
            print(f"step {step}: no finite predictions at all, skipped")
            continue
        xm = np.asarray(sample.node_x_mm, dtype=np.float64) * 1e-3
        ym = np.asarray(sample.node_y_mm, dtype=np.float64) * 1e-3
        valid = mesh_validity_mask(xm, ym, mesh.tri, exclude=(band | ~finite))
        op = build_p1_curl_operator(xm, ym, mesh.tri, valid)
        a_fit, rel = fit_nodal_a(op, b_pred[op.element_index])
        # gauge anchor: A = 0 on the outer stator arc, matching the solver's Dirichlet
        r_mm = np.hypot(sample.node_x_mm, sample.node_y_mm)
        outer = r_mm > float(r_mm.max()) - 0.2
        a_fit = a_fit - float(a_fit[outer].mean())
        note = ""
        if args.gap_harmonic > 0:
            a_fit, harm_rel, n_orphan = _gap_harmonic_patch(
                a_fit, op, b_pred, mesh, sample, args.gap_harmonic)
            note = f"  harm-fit {harm_rel:.4f} on {n_orphan} orphan nodes"
        out[f"step_{step}"] = a_fit.astype(np.float64)
        residuals.append(rel)
        print(f"step {step:2d}  lsqr rel-resid {rel:.4f}  fit on {op.n_valid}/{finite.size} elems  "
              f"|A| max {np.abs(a_fit).max():.4e}  {time.perf_counter()-t0:.1f}s{note}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **out)
    print(f"wrote {args.out} with {len(out)} steps; "
          f"mean lsqr residual {np.mean(residuals):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
