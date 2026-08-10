"""Regression: the eMach torque_operators port must reproduce fem_warmstart's numbers.

The eMach package is solver-agnostic by design, so it cannot be validated on
Motor-CAD data from inside eMach. This script closes that loop from the side that
has the data: build the eMach mesh/material objects from a `FemDomain`, run both
operators through the eMach code path, and compare against the in-repo
implementation that was measured at +0.475% (spread 0.024 pp) vs Arkkio.

    python results/logs/vw_emach_port_check.py --emach D:/KDH/eMach_torque_wt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from eval.doe_dataset import load_doe_cases                          # noqa: E402
from eval.torque import arkkio_torque as arkkio_ref                  # noqa: E402
from eval.torque import build_airgap_band                            # noqa: E402
from fem_warmstart.domain import STEEL                               # noqa: E402
from fem_warmstart.torque_vw import (                                # noqa: E402
    build_virtual_displacement,
    domain_and_a,
    torque_vw_coulomb,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--emach", type=Path, default=Path("D:/KDH/eMach_torque_wt"),
                    help="checkout containing tools/torque_operators")
    ap.add_argument("--data-dir", default="backup/doe_data_v3")
    ap.add_argument("--case", type=int, default=4)
    ap.add_argument("--steps", type=int, nargs="+", default=[5, 12, 20, 28, 36])
    ap.add_argument("--tol-pct", type=float, default=1e-6,
                    help="max allowed relative difference between the two code paths")
    args = ap.parse_args()

    sys.path.insert(0, str(args.emach))
    from tools.torque_operators import (                             # noqa: E402
        BHCurve, ElementMaterials, TriMesh, coulomb_torque,
        element_b_and_area, radial_blend,
    )

    manifest = json.loads((Path(args.data_dir) / "doe_manifest.json").read_text(encoding="utf-8"))
    rec = load_doe_cases(manifest, args.data_dir, case_indices=[args.case]).records[0]
    me = rec.mesh
    band = build_airgap_band(me.node_x_mm, me.node_y_mm, me.tri, me.reg_code,
                             me.name_of_code, moving_reg_codes=me.moving_reg_codes)
    print(f"case {args.case}: band {band.n_elements} elements, "
          f"symmetry x{band.symmetry_multiplier}")

    worst = 0.0
    rows = []
    for step in args.steps:
        dom, a_full, _ = domain_and_a(rec, step)

        # in-repo path
        ref = torque_vw_coulomb(dom, a_full, symmetry_multiplier=band.symmetry_multiplier)

        # eMach path: same domain, rebuilt through the solver-agnostic containers
        mesh = TriMesh(x=dom.x, y=dom.y, tri=dom.tri)
        curves = tuple(BHCurve(c.H, c.B) for c in dom.curves)
        cidx = np.where(dom.material == STEEL, dom.curve_index, -1).astype(np.int64)
        mats = ElementMaterials(nu_linear=dom.nu_linear, br=dom.br,
                                curve_index=cidx, curves=curves)
        vd = build_virtual_displacement(dom)
        disp = radial_blend(mesh, vd.r_inner_m, vd.r_outer_m)
        got = coulomb_torque(mesh, mats, a_full, disp,
                             symmetry_multiplier=band.symmetry_multiplier,
                             self_test=True)

        # in-repo returns dW'/dtheta directly; eMach negates to the +theta
        # convention that arkkio_torque uses. Compare on magnitude and record both.
        d = 100.0 * abs(abs(got["torque"]) - abs(ref["torque"])) / abs(ref["torque"])
        worst = max(worst, d)
        b_from_a = dom.to_export_elements(dom.element_b(a_full), fill=0.0)
        t_ark = arkkio_ref(band, b_from_a[:, 0], b_from_a[:, 1])
        vs_ark = 100.0 * (got["torque"] - t_ark) / abs(t_ark)
        rows.append({"step": step, "emach": got["torque"], "in_repo": ref["torque"],
                     "arkkio": t_ark, "port_diff_pct": d, "vs_arkkio_pct": vs_ark,
                     "profile_invariance_pct": got.get("profile_invariance_pct")})
        print(f"  step {step:3d}  eMach {got['torque']:10.3f}   in-repo {ref['torque']:10.3f}"
              f"   port diff {d:.3e}%   vs Arkkio {vs_ark:+6.3f}%"
              f"   prof-inv {got.get('profile_invariance_pct', float('nan')):.4f}%")

    ok = worst <= args.tol_pct
    print(f"\nworst port difference {worst:.3e}%  (tolerance {args.tol_pct:g}%)")
    print(f"mean vs Arkkio {np.mean([r['vs_arkkio_pct'] for r in rows]):+.3f}%, "
          f"spread {np.ptp([r['vs_arkkio_pct'] for r in rows]):.3f} pp")
    out = Path("results/vw_emach_port_check.json")
    out.write_text(json.dumps({"case": args.case, "rows": rows,
                               "worst_port_diff_pct": worst}, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    print("EMACH_PORT_" + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
