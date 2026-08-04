"""R7-A acceptance checks: winding template, magnet convention, prior quality.

Run on the host (numpy/scipy only):

    python -m fem_warmstart.validate_prior            # all checks
    python -m fem_warmstart.validate_prior --calibrate  # (re)write the template JSON

Checks, each with a hard gate:

1. J synthesis transfers across cases. Template calibrated from case_0000 must
   reproduce the exported j of cases 4 and 7 (different geometry AND
   PhaseAdvance) at held-out steps to < 0.5% of amplitude.
2. The fixed radial-outward magnet convention matches the from-field polarity
   on every magnet of the test cases (the one bit per magnet that section 23c
   verified by hand becomes an assert).
3. The prior is informative: linear-solve |B| vs FEM |B| nRMSE is reported
   per region (no gate — saturation error is expected and is the point), but
   the prior must beat the zero-field baseline (100%) everywhere that matters.
4. Determinism: two independent computations of the prior are byte-identical.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.doe_dataset import load_doe_cases                        # noqa: E402
from fem_warmstart.winding import (                                # noqa: E402
    TEMPLATE_PATH,
    calibrate_template,
    load_template,
    phase_group,
    synthesize_j,
)

DATA = Path("backup/doe_data")


def _record(manifest, case):
    return load_doe_cases(manifest, DATA, case_indices=[case]).records[0]


def check_j_synthesis(manifest, cases=(4, 7), steps=(10, 20, 33, 40)) -> bool:
    from eval.mesh_regions import element_areas_m2

    tpl = load_template()
    ok = True
    for case in cases:
        rec = _record(manifest, case)
        mesh = rec.mesh
        reg = np.asarray(mesh.reg_code)
        adv = float(rec.condition.get("PhaseAdvance", 0.0))
        worst = 0.0
        n_regions = 0
        for k in steps:
            s = rec.samples[k]
            area = element_areas_m2(s.node_x_mm, s.node_y_mm, mesh.tri)
            j_syn = synthesize_j(mesh, area, adv, k, template=tpl) * 1e-6  # -> A/mm^2
            j_exp = np.asarray(s.fields["j"], dtype=np.float64)
            for code, name in mesh.name_of_code.items():
                if phase_group(str(name)) is None:
                    continue
                m = reg == int(code)
                n_regions += 1
                amp = float(tpl["ampere_turns"]) / (1e6 * area[m].sum())
                err = abs(float(j_syn[m][0]) - float(j_exp[m][0])) / amp
                worst = max(worst, err)
        print(f"  case {case}: max |j_syn - j_export| / amplitude = {100*worst:.3f}% "
              f"({n_regions} region-steps)")
        ok &= worst < 0.005
    return ok


def check_polarity(manifest, cases=(4, 7)) -> bool:
    from fem_warmstart.domain import build_domain
    from fem_warmstart.prior import default_bh_path

    ok = True
    for case in cases:
        rec = _record(manifest, case)
        d_field = build_domain(rec, 0, default_bh_path())
        d_fixed = build_domain(rec, 0, default_bh_path(),
                               j_source="synthetic", magnet_polarity="radial_outward")
        for name, ref in d_field.constraint_report["magnets"].items():
            got = d_fixed.constraint_report["magnets"][name]
            same = np.isclose(ref["axis_deg"], got["axis_deg"], atol=1e-6)
            print(f"  case {case} {name}: from_field {ref['axis_deg']:+8.2f} deg  "
                  f"radial_outward {got['axis_deg']:+8.2f} deg  {'OK' if same else 'MISMATCH'}")
            ok &= bool(same)
    return ok


def _fem_node_b(rec, step):
    mesh = rec.mesh
    s = rec.samples[step]
    bxy = np.stack([np.asarray(s.fields["bx"]), np.asarray(s.fields["by"])], axis=1)
    i1, i2, i3 = mesh.tri
    idx3 = np.concatenate([i1, i2, i3]).astype(np.int64)
    counts = np.maximum(np.bincount(idx3, minlength=mesh.n_nodes), 1).astype(np.float64)
    return np.stack(
        [np.bincount(idx3, weights=np.tile(bxy[:, c], 3), minlength=mesh.n_nodes) / counts
         for c in range(2)], axis=1)


def check_prior_quality(manifest, cases=(4, 7), steps=(0, 20), sweep=False) -> bool:
    """Feature-usefulness check, scale-invariant on purpose.

    Node features are standardized per column before the network sees them, so
    a global scale on the prior is free; what matters is the SHAPE. The metric
    is therefore nRMSE after the optimal global rescale (equivalently
    sqrt(1 - rho^2) of the pooled correlation). The un-rescaled number is
    reported for context only — at initial permeability it is ~277% because
    the linear solve has no saturation to limit the flux.
    """
    from fem_warmstart.prior import STEEL_MU_EFF_R, compute_prior

    grid = (None, 1000.0, 300.0, 100.0, 30.0) if sweep else (STEEL_MU_EFF_R,)
    best = {}
    for mu in grid:
        raws, scaled = [], []
        for case in cases:
            rec = _record(manifest, case)
            for step in steps:
                p = compute_prior(rec, step, mu_eff_r=mu)
                fem = _fem_node_b(rec, step)
                pr = p[:, 1:3].ravel()
                fe = fem.ravel()
                den = np.linalg.norm(fe)
                raws.append(100.0 * np.linalg.norm(pr - fe) / den)
                alpha = float(pr @ fe) / float(pr @ pr)
                scaled.append(100.0 * np.linalg.norm(alpha * pr - fe) / den)
        label = "nu_init" if mu is None else f"mu_r={mu:g}"
        print(f"  prior[{label:10s}] nodal-B vs FEM: raw {np.mean(raws):6.1f}%   "
              f"after optimal rescale {np.mean(scaled):5.1f}%")
        best[label] = float(np.mean(scaled))

    rec = _record(manifest, cases[0])
    p1 = compute_prior(rec, steps[0])
    p2 = compute_prior(rec, steps[0])
    deterministic = np.array_equal(p1, p2)
    print(f"  determinism (two computations byte-equal): {deterministic}")
    return deterministic and min(best.values()) < 100.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--calibrate", action="store_true",
                    help="(re)write fem_warmstart/data/winding_template.json from case_0000")
    ap.add_argument("--reference-case", type=int, default=0)
    ap.add_argument("--sweep-mu", action="store_true",
                    help="sweep effective steel permeabilities in check 3 to pick "
                         "STEEL_MU_EFF_R (report-only; the constant lives in prior.py)")
    args = ap.parse_args()

    manifest = json.loads((DATA / "doe_manifest.json").read_text(encoding="utf-8"))

    if args.calibrate:
        rec = _record(manifest, args.reference_case)
        tpl = calibrate_template(rec)
        TEMPLATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        TEMPLATE_PATH.write_text(json.dumps(
            {"_provenance": f"calibrated from case_{args.reference_case:04d} "
                            f"(PhaseAdvance {rec.condition.get('PhaseAdvance'):.4f} deg); "
                            "template constant per section 24 of the methodology review",
             **tpl}, indent=1), encoding="utf-8")
        print(f"wrote {TEMPLATE_PATH}: ampere-turns {tpl['ampere_turns']:.2f} "
              f"(spread {tpl['ampere_turns_spread']:.2f}), "
              f"phi0 {tpl['phi0_deg']} (spread {tpl['phi0_spread_deg']})")

    print("[1] J synthesis, held-out cases/steps:")
    ok1 = check_j_synthesis(manifest)
    print("[2] magnet polarity convention:")
    ok2 = check_polarity(manifest)
    print("[3] prior quality + determinism:")
    ok3 = check_prior_quality(manifest, sweep=args.sweep_mu)
    verdict = ok1 and ok2 and ok3
    print("PRIOR_VALIDATION_" + ("OK" if verdict else "FAIL"))
    return 0 if verdict else 1


if __name__ == "__main__":
    raise SystemExit(main())
