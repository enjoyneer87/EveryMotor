"""Solve a whole rotor sweep and benchmark Newton initial guesses.

Two things at once:

1. **Solver validation across rotor positions.** Every step is solved from a cold
   start and its Arkkio torque is compared with the torque of Motor-CAD's own
   exported field on the same band, so the check is the operator the campaign
   already trusts (methodology review section 8) rather than a bespoke norm.

2. **Warm-start benchmark.** The same problem is solved from each initial guess
   with identical tolerance and damping, and only the guess changes:

   * ``zero``      -- a = 0, the lower baseline;
   * ``previous``  -- the converged solution of the preceding rotor step, which
     is what a production sweep actually does and therefore the bar an AI
     initialisation has to clear;
   * ``ai``        -- nodal A predicted by the curl-A surrogate, supplied as an
     .npz of per-step nodal vectors (``--ai-init``). Optional: without it the
     first two are still reported.

Usage:
    python -m fem_warmstart.run_sweep --case 4 --steps 45 --out results/fem_warmstart_case4.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.doe_dataset import load_doe_cases                     # noqa: E402
from eval.torque import arkkio_torque, build_airgap_band        # noqa: E402
from fem_warmstart.domain import build_domain                   # noqa: E402
from fem_warmstart.newton import newton_solve                   # noqa: E402

DEFAULT_BH = ("D:/KDH/Sim_4SolverX/DOE_Ext/case_0000/TestCAD1/FEResultsData/"
              "Steel_Material_BH_Magnetic_Properties_Autofile.bh")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", type=Path, default=Path("backup/doe_data"))
    ap.add_argument("--case", type=int, default=4)
    ap.add_argument("--steps", type=int, default=45)
    ap.add_argument("--bh", default=DEFAULT_BH)
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--axial-length-m", type=float, default=0.150)
    ap.add_argument("--ai-init", type=Path, default=None,
                    help=".npz with arrays named step_<k> of nodal A predictions")
    ap.add_argument("--out", type=Path, default=Path("results/fem_warmstart_sweep.json"))
    args = ap.parse_args()

    manifest = json.loads((args.data_dir / "doe_manifest.json").read_text(encoding="utf-8"))
    record = load_doe_cases(manifest, args.data_dir, case_indices=[args.case]).records[0]
    mesh = record.mesh
    n_steps = min(args.steps, len(record.samples))

    ai = np.load(args.ai_init) if args.ai_init else None

    rows = []
    previous = None
    for step in range(n_steps):
        sample = record.samples[step]
        domain = build_domain(record, step, args.bh)
        band = build_airgap_band(
            sample.node_x_mm, sample.node_y_mm, mesh.tri, mesh.reg_code,
            mesh.name_of_code, moving_reg_codes=mesh.moving_reg_codes)
        bx_fem = np.asarray(sample.fields["bx"], dtype=np.float64)
        by_fem = np.asarray(sample.fields["by"], dtype=np.float64)
        t_fem = float(arkkio_torque(band, bx_fem, by_fem, args.axial_length_m))

        guesses = {"zero": None,
                   "previous": None if previous is None else domain.lift_guess(previous)}
        if ai is not None and f"step_{step}" in ai:
            guesses["ai"] = domain.lift_guess(np.asarray(ai[f"step_{step}"], dtype=np.float64))

        row = {"step": step, "torque_fem": t_fem,
               "rotate_step": float(sample.rotate_step),
               "n_dof": domain.n_dof, "n_elements": domain.n_elements}
        solutions = {}
        for name, a0 in guesses.items():
            res = newton_solve(domain, a0=a0, tol=args.tol)
            solutions[name] = res
            row[name] = {
                "iterations": res.iterations,
                "converged": res.converged,
                "wall_time_s": round(res.wall_time_s, 3),
                "initial_residual": res.residual_history[0] if res.residual_history else None,
            }

        ref = solutions.get("previous") or solutions["zero"]
        b_solved = domain.to_export_elements(domain.element_b(ref.a_nodal), fill=0.0)
        t_me = float(arkkio_torque(band, b_solved[:, 0], b_solved[:, 1], args.axial_length_m))
        row["torque_poc"] = t_me
        row["torque_error_pct"] = 100.0 * (t_me - t_fem) / abs(t_fem) if t_fem else None

        # cross-check that every guess reached the same solution
        spread = max(
            float(np.max(np.abs(s.a_nodal[:domain.n_original_nodes]
                                - ref.a_nodal[:domain.n_original_nodes])))
            for s in solutions.values())
        row["solution_spread"] = spread

        rows.append(row)
        parts = "  ".join(f"{k} {row[k]['iterations']:2d}it" for k in guesses)
        print(f"step {step:2d}  {parts}  torque {t_me:9.2f} vs FEM {t_fem:9.2f} "
              f"({row['torque_error_pct']:+.3f}%)  spread {spread:.2e}", flush=True)
        previous = ref.a_nodal[:domain.n_original_nodes].copy()

    summary = {}
    for name in ("zero", "previous", "ai"):
        its = [r[name]["iterations"] for r in rows if name in r]
        if not its:
            continue
        summary[name] = {
            "n": len(its),
            "mean_iterations": float(np.mean(its)),
            "median_iterations": float(np.median(its)),
            "total_iterations": int(np.sum(its)),
            "mean_wall_time_s": float(np.mean([r[name]["wall_time_s"] for r in rows if name in r])),
            "all_converged": bool(all(r[name]["converged"] for r in rows if name in r)),
        }
    err = np.array([r["torque_error_pct"] for r in rows], dtype=float)
    summary["torque_vs_motorcad_pct"] = {
        "mean": float(np.mean(err)), "max_abs": float(np.max(np.abs(err))),
        "rms": float(np.sqrt(np.mean(err ** 2))),
    }

    out = {"case": args.case, "n_steps": n_steps, "tol": args.tol,
           "axial_length_m": args.axial_length_m, "summary": summary, "steps": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("\n" + json.dumps(summary, indent=1))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
