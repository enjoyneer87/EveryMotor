"""How well can harmonics describe the air-gap field the G1' gate is scored on?

Section 25 retired the original G1 by measuring that it sat below the curl
representation floor. This asks the same kind of question about the air-gap leg
of its replacement: if a large share of the target's energy is element-level
scatter that no smooth angular description can express, then part of the
remaining 1.618 pp to G1' is a property of the metric, not of the model.

Method. On EXACTLY the elements the gate scores -- region `a<k>`, sliding band
excluded, geometrically invalid elements excluded (`eval.benchmark` rules) -- fit
the field's (Br, Btheta) to angular harmonics, rebuild (Bx, By), and score the
reconstruction as if it were a model, with the gate's own statistic: pooled
nRMSE of |B|. That makes the number directly comparable to the champion's
airgap 6.618%.

Three bases, because the gap between them localises the cause:
  admissible   only the orders an anti-periodic one-pole sector permits (4*odd).
               Everything outside is discretisation by construction.
  n <= 200     lets forbidden angular orders in too.
  n <= 350     ~700 parameters on ~750 elements: essentially interpolation in
               theta. Whatever this still misses is radial variation across the
               band plus element scatter, and no angular scheme can recover it.

This is a DESCRIPTIVE bound, not a hard floor like the representation floors.
The scatter is deterministic given the mesh, and the surrogate sees the mesh, so
in principle it is learnable. Read it as "how much of the target is not smooth
in theta", which is what a Fourier-guided or band-limited approach would discard.

    python results/logs/airgap_smoothness_floor.py --emach D:/KDH/eMach_torque_wt
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
from eval.mesh_regions import AIRGAP_NAME_PATTERN, sliding_band_mask  # noqa: E402
from phase1_static.discrete_curl import mesh_validity_mask        # noqa: E402

LEGACY_TEST = [4, 7, 18, 32, 37, 39]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--emach", type=Path, default=Path("D:/KDH/eMach_torque_wt"))
    ap.add_argument("--data-dir", default="backup/doe_data_v3")
    ap.add_argument("--cases", type=int, nargs="+", default=LEGACY_TEST)
    ap.add_argument("--pole-pairs", type=int, default=4)
    ap.add_argument("--step-stride", type=int, default=1)
    ap.add_argument("--out", type=Path, default=Path("results/airgap_smoothness_floor.json"))
    args = ap.parse_args()

    sys.path.insert(0, str(args.emach))
    from tools.torque_operators import admissible_orders, harmonic_residual  # noqa: E402

    bases = {
        "admissible_4odd": list(admissible_orders(args.pole_pairs)),
        "all_n_le_200": list(range(1, 201)),
        "all_n_le_350": list(range(1, 351)),
    }

    manifest = json.loads((Path(args.data_dir) / "doe_manifest.json").read_text(encoding="utf-8"))
    # Pooled accumulators: sum of squared error and of squared truth, over every
    # scored element of every step of every case -- the gate's own aggregation.
    acc = {k: {"se": 0.0, "st": 0.0, "n": 0} for k in bases}
    per_case = {}
    t0 = time.time()

    for ci in args.cases:
        rec = load_doe_cases(manifest, args.data_dir, case_indices=[ci]).records[0]
        me = rec.mesh
        is_gap = np.array([bool(AIRGAP_NAME_PATTERN.match(str(me.name_of_code.get(int(c), ""))))
                           for c in np.asarray(me.reg_code)])
        band = sliding_band_mask(me.reg_code, me.name_of_code, me.moving_reg_codes)
        cacc = {k: {"se": 0.0, "st": 0.0, "n": 0} for k in bases}

        for si in range(0, len(rec.samples), args.step_stride):
            s = rec.samples[si]
            valid = mesh_validity_mask(s.node_x_mm * 1e-3, s.node_y_mm * 1e-3,
                                       me.tri, exclude=band)
            sel = np.flatnonzero(is_gap & valid)
            if sel.size < 50:
                continue
            tri = np.asarray(me.tri)[:, sel]
            cx = s.node_x_mm[tri].mean(0) * 1e-3
            cy = s.node_y_mm[tri].mean(0) * 1e-3
            r = np.hypot(cx, cy)
            ct, st_ = cx / r, cy / r
            phi = np.arctan2(st_, ct)

            bx = np.asarray(s.fields["bx"])[sel]
            by = np.asarray(s.fields["by"])[sel]
            b_r = bx * ct + by * st_
            b_t = -bx * st_ + by * ct
            b_true = np.hypot(bx, by)

            for name, orders in bases.items():
                fr = harmonic_residual(phi, b_r, orders)["reconstruction"]
                ft = harmonic_residual(phi, b_t, orders)["reconstruction"]
                # back to cartesian, then score |B| exactly as the gate does
                rx = fr * ct - ft * st_
                ry = fr * st_ + ft * ct
                b_rec = np.hypot(rx, ry)
                d = b_rec - b_true
                for tgt in (acc[name], cacc[name]):
                    tgt["se"] += float(d @ d)
                    tgt["st"] += float(b_true @ b_true)
                    tgt["n"] += d.size

        row = {k: 100.0 * np.sqrt(v["se"] / v["n"]) / np.sqrt(v["st"] / v["n"])
               for k, v in cacc.items()}
        per_case[str(ci)] = row
        print(f"  case {ci:3d}  " + "  ".join(f"{k} {row[k]:6.3f}%" for k in bases)
              + f"   [{time.time()-t0:.0f}s]")

    pooled = {k: 100.0 * np.sqrt(v["se"] / v["n"]) / np.sqrt(v["st"] / v["n"])
              for k, v in acc.items()}
    print("\nPOOLED |B| nRMSE of the harmonic reconstruction, gate element set")
    for k in bases:
        print(f"  {k:16s} {pooled[k]:6.3f}%   ({len(bases[k])} orders, "
              f"{2*len(bases[k])} params)")
    print(f"\n  champion airgap (v2, legacy 6-test)   6.618%")
    print(f"  G1' airgap target                     5.000%")

    out = {
        "note": ("pooled |B| nRMSE of an angular-harmonic reconstruction scored on the "
                 "gate's own airgap element set; descriptive bound, not a hard floor"),
        "cases": args.cases, "pole_pairs": args.pole_pairs,
        "n_elements_pooled": {k: acc[k]["n"] for k in bases},
        "pooled_pct": pooled, "per_case_pct": per_case,
        "reference": {"champion_airgap_pct": 6.618, "g1prime_airgap_target_pct": 5.0},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
