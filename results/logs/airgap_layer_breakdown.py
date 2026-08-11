"""Where in the air gap does the non-smooth content actually live?

Section 31 measured that the best angular description of the gap field leaves a
large residual, and section 31a read that as physics. Section 31c then showed the
refinement test cannot separate physics from moving/stationary interface mapping
error -- both predict the same signature -- and that on one case/step the residual
is extremely localised: the stationary stator-side layer is rough while the
mid-gap layers are five times smoother.

This runs that layer decomposition over the whole legacy holdout so the pattern
is established rather than anecdotal. Per air-gap layer, per case, per rotor step:
fit (Br, Bt) to angular harmonics, rebuild, and score with the gate's statistic
(pooled |B| nRMSE), so the numbers are comparable to the 6.618% champion figure.

Layers are reported with their mean radius and whether the sliding-band mask
excludes them -- the gate scores only what survives that mask, and knowing which
layer that is turns out to matter more than any aggregate.

    python results/logs/airgap_layer_breakdown.py --emach D:/KDH/eMach_torque_wt
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from eval.doe_dataset import load_doe_cases                           # noqa: E402
from eval.mesh_regions import AIRGAP_NAME_PATTERN, sliding_band_mask  # noqa: E402
from phase1_static.discrete_curl import mesh_validity_mask            # noqa: E402

LEGACY_TEST = [4, 7, 18, 32, 37, 39]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--emach", type=Path, default=Path("D:/KDH/eMach_torque_wt"))
    ap.add_argument("--data-dir", default="backup/doe_data_v3")
    ap.add_argument("--cases", type=int, nargs="+", default=LEGACY_TEST)
    ap.add_argument("--pole-pairs", type=int, default=4)
    ap.add_argument("--step-stride", type=int, default=1)
    ap.add_argument("--out", type=Path, default=Path("results/airgap_layer_breakdown.json"))
    args = ap.parse_args()

    sys.path.insert(0, str(args.emach))
    from tools.torque_operators import admissible_orders, harmonic_residual  # noqa: E402

    bases = {"admissible_4odd": list(admissible_orders(args.pole_pairs)),
             "all_n_le_200": list(range(1, 201))}

    manifest = json.loads((Path(args.data_dir) / "doe_manifest.json").read_text(encoding="utf-8"))
    # accumulators keyed (layer, basis) -> pooled squared error / squared truth
    acc: dict = {}
    meta: dict = {}
    t0 = time.time()

    for ci in args.cases:
        rec = load_doe_cases(manifest, args.data_dir, case_indices=[ci]).records[0]
        me = rec.mesh
        codes = np.asarray(me.reg_code)
        band = sliding_band_mask(me.reg_code, me.name_of_code, me.moving_reg_codes)
        layers = {}
        for c in sorted({int(x) for x in codes}):
            nm = str(me.name_of_code.get(c, ""))
            if AIRGAP_NAME_PATTERN.match(nm):
                layers[nm] = codes == c

        for si in range(0, len(rec.samples), args.step_stride):
            s = rec.samples[si]
            valid = mesh_validity_mask(s.node_x_mm * 1e-3, s.node_y_mm * 1e-3,
                                       me.tri, exclude=band)
            for nm, mask in layers.items():
                sel = np.flatnonzero(mask)
                if sel.size < 50:
                    continue
                tri = np.asarray(me.tri)[:, sel]
                cx = s.node_x_mm[tri].mean(0) * 1e-3
                cy = s.node_y_mm[tri].mean(0) * 1e-3
                r = np.hypot(cx, cy)
                ct, st = cx / r, cy / r
                phi = np.arctan2(st, ct)
                bx = np.asarray(s.fields["bx"])[sel]
                by = np.asarray(s.fields["by"])[sel]
                b_r, b_t = bx * ct + by * st, -bx * st + by * ct
                b_true = np.hypot(bx, by)

                m = meta.setdefault(nm, {"n_elem": int(sel.size), "r_mid_mm": [],
                                         "in_sliding_band": bool(band[sel].mean() > 0.5),
                                         "scored_by_gate": bool(valid[sel].mean() > 0.5)})
                m["r_mid_mm"].append(float(1e3 * r.mean()))

                for bname, orders in bases.items():
                    fr = harmonic_residual(phi, b_r, orders)["reconstruction"]
                    ft = harmonic_residual(phi, b_t, orders)["reconstruction"]
                    rx, ry = fr * ct - ft * st, fr * st + ft * ct
                    d = np.hypot(rx, ry) - b_true
                    a = acc.setdefault((nm, bname), {"se": 0.0, "st": 0.0, "n": 0})
                    a["se"] += float(d @ d)
                    a["st"] += float(b_true @ b_true)
                    a["n"] += d.size
        print(f"  case {ci:3d} done  [{time.time()-t0:.0f}s]", flush=True)

    rows = []
    for nm in sorted(meta, key=lambda k: np.mean(meta[k]["r_mid_mm"])):
        m = meta[nm]
        row = {"layer": nm, "r_mid_mm": float(np.mean(m["r_mid_mm"])),
               "n_elem_case4": m["n_elem"],
               "in_sliding_band": m["in_sliding_band"],
               "scored_by_gate": m["scored_by_gate"]}
        for bname in bases:
            a = acc[(nm, bname)]
            row[bname] = 100.0 * np.sqrt(a["se"] / a["n"]) / np.sqrt(a["st"] / a["n"])
        rows.append(row)

    print(f"\nPer-layer |B| nRMSE of the angular-harmonic reconstruction "
          f"({len(args.cases)} geometries x {len(range(0, 45, args.step_stride))} steps)")
    print(f"{'layer':6s} {'r_mid[mm]':>10s} {'slidingband':>12s} {'gate-scored':>12s} "
          f"{'admissible':>11s} {'n<=200':>9s}")
    for r in rows:
        print(f"{r['layer']:6s} {r['r_mid_mm']:10.3f} {str(r['in_sliding_band']):>12s} "
              f"{str(r['scored_by_gate']):>12s} {r['admissible_4odd']:10.3f}% "
              f"{r['all_n_le_200']:8.3f}%")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"cases": args.cases, "pole_pairs": args.pole_pairs, "layers": rows,
         "note": ("pooled |B| nRMSE of a per-layer angular-harmonic reconstruction; "
                  "the gate scores only layers that survive the sliding-band mask")},
        indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
