"""Cross-check the three torque operators on the SOLVER'S OWN field.

Arkkio (Maxwell stress, annulus-averaged) vs Coulomb local virtual work (one
solve) vs global virtual work (finite difference over two solves), all fed the
exported FEM field of the same case and step, all reported for the full machine.

The question this answers is the one section 22 currently answers by assertion:
the 0.35% Arkkio-vs-Motor-CAD residual is attributed to mesh faceting of the
band area. If an independent operator on the same field lands in the same place,
the residual is in the field/mesh; if virtual work sits closer to Motor-CAD, it
is Arkkio's band discretisation.

    python results/logs/vw_torque_compare.py --case 4 --steps 5 15 25 35
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from eval.doe_dataset import load_doe_cases                       # noqa: E402
from eval.torque import arkkio_torque, build_airgap_band          # noqa: E402
from fem_warmstart.torque_vw import (                             # noqa: E402
    coenergy,
    domain_and_a,
    torque_vw_coulomb,
    torque_vw_finite_difference,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", default="backup/doe_data_v3")
    ap.add_argument("--case", type=int, default=4)
    ap.add_argument("--steps", type=int, nargs="+", default=[5, 15, 25, 35])
    ap.add_argument("--axial-length-m", type=float, default=1.0,
                    help="1.0 = torque per metre of stack, the campaign convention")
    ap.add_argument("--skip-fd", action="store_true",
                    help="skip the finite-difference arm (2 extra domain builds per step)")
    ap.add_argument("--out", type=Path, default=Path("results/vw_torque_compare.json"))
    args = ap.parse_args()

    manifest = json.loads((Path(args.data_dir) / "doe_manifest.json").read_text(encoding="utf-8"))
    rec = load_doe_cases(manifest, args.data_dir, case_indices=[args.case]).records[0]
    print(f"case {args.case}: {rec.mesh.n_nodes} nodes, {len(rec.samples)} steps, "
          f"Ipk {float(rec.condition.get('PeakCurrent', float('nan'))):.1f} A")

    mesh = rec.mesh
    band = build_airgap_band(mesh.node_x_mm, mesh.node_y_mm, mesh.tri, mesh.reg_code,
                             mesh.name_of_code, moving_reg_codes=mesh.moving_reg_codes)
    print(f"Arkkio band: {band.n_elements} elements, thickness {1e3*band.thickness_m:.3f} mm, "
          f"symmetry x{band.symmetry_multiplier}")

    rows = []
    for step in args.steps:
        t0 = time.time()
        s = rec.samples[step]

        # --- Arkkio on the exported element field --------------------------------
        t_ark = arkkio_torque(band, np.asarray(s.fields["bx"]), np.asarray(s.fields["by"]),
                              axial_length_m=args.axial_length_m)

        # --- Coulomb local virtual work, one solve -------------------------------
        # The export has no nodal A (its `a` is element-wise), so both operators
        # are scored on the reference solver's converged nodal field instead.
        dom, a_full, nwt = domain_and_a(rec, step)
        cou = torque_vw_coulomb(dom, a_full, axial_length_m=args.axial_length_m,
                                self_test=True)

        # Arkkio on the SAME field, so this is an operator comparison rather than
        # a field comparison.
        b_from_a = dom.to_export_elements(dom.element_b(a_full), fill=0.0)
        t_ark_a = arkkio_torque(band, b_from_a[:, 0], b_from_a[:, 1],
                                axial_length_m=args.axial_length_m)

        row = {
            "step": step,
            "arkkio_exported_B": t_ark,
            "arkkio_from_nodal_A": t_ark_a,
            "coulomb_vw": cou["torque"],
            "coulomb_profile_invariance_pct": cou.get("profile_invariance_pct"),
            "coenergy_J_per_m": cou["coenergy_J_per_m"],
            "newton": nwt,
        }

        if not args.skip_fd:
            fd = torque_vw_finite_difference(rec, step, axial_length_m=args.axial_length_m)
            row["global_vw_fd"] = fd["torque"]
            row["d_theta_mech_deg"] = float(np.degrees(fd["d_theta_mech_rad"]))

        # One convention for the printout. `arkkio_torque` returns torque about
        # +theta; this rotor motors clockwise, so its value is negative and
        # tools/compare_torque_waveforms.py already negates it to reach Motor-CAD's
        # motoring-positive convention. Coulomb comes out motoring-positive
        # directly. Negating Arkkio is what puts them on one axis -- it is not a
        # fudge, it is the same flip that figure already documents.
        row["arkkio_motoring"] = -t_ark_a
        row["coulomb_vs_arkkio_pct"] = 100.0 * (cou["torque"] - (-t_ark_a)) / abs(t_ark_a)
        rows.append(row)
        print(f"  step {step:3d}  motoring convention:  Arkkio(B_exp) {-t_ark:10.2f}   "
              f"Arkkio(curl A) {-t_ark_a:10.2f}   Coulomb {cou['torque']:10.2f}"
              f"  ({row['coulomb_vs_arkkio_pct']:+6.2f}% vs Arkkio)"
              + (f"   VW-FD {row['global_vw_fd']:9.2f}" if not args.skip_fd else "")
              + f"   [{time.time()-t0:.1f}s]")
        if cou.get("profile_invariance_pct") is not None:
            print(f"           virtual-displacement profile invariance: "
                  f"{cou['profile_invariance_pct']:.4f}%  "
                  f"(linear vs smoothstep; should be ~0)")

    summary = {
        "case": args.case,
        "data_dir": str(args.data_dir),
        "axial_length_m": args.axial_length_m,
        "band": band.describe(),
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")

    # A single headline: how far the independent operator sits from Arkkio.
    d = [r["coulomb_vs_arkkio_pct"] for r in rows]
    print(f"Coulomb VW vs Arkkio MST, same field, same mesh: "
          f"mean {np.mean(d):+.3f}%, spread {np.max(d)-np.min(d):.3f} pp")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
