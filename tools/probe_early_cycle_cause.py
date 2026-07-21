#!/usr/bin/env python3
"""Intervention tests separating feature artifact from model limitation.

`diagnose_early_cycle_error.py` showed the early-cycle torque error is neither
universal (per-case ratios span 0.39 to 4.44) nor confined to holdout cases
(train case 3 shows 3.03). This script runs the interventions that tell the
remaining hypotheses apart:

1. **time_s sensitivity** — perturb only `time_s` and measure how much the
   prediction moves. If the model barely responds, the step-0 `time_s` value
   cannot be causing anything.
2. **Angle-encoding aliasing** — `rotor_angle_features` has a period of two
   sectors (90 mech deg), so cum = 0 and cum = -88 give almost identical angle
   inputs while the rotor sits 88 deg apart in the (unwrapped) geometry. Compare
   the model's error at those two steps.
3. **Train-set fit** — if the model cannot fit the *training* torque at early
   steps either, the limitation is capacity/representation, not generalization.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from eval.case_split import load_case_split  # noqa: E402
from eval.doe_dataset import load_doe_cases  # noqa: E402
from eval.torque import arkkio_torque, build_airgap_band  # noqa: E402


def torque_of(record, band, predictor, sample, axial):
    pred = predictor.predict(record, sample)
    return arkkio_torque(band, pred[:, 0], pred[:, 1], axial)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=Path("backup/doe_data"))
    parser.add_argument("--ckpt", type=Path, default=Path("results/mgn_nodeB_long.pt"))
    parser.add_argument("--split", type=Path, default=Path("eval/splits/doe40_case_split.json"))
    parser.add_argument("--axial-length-m", type=float, default=0.150)
    args = parser.parse_args()

    manifest = json.loads((args.data_dir / "doe_manifest.json").read_text(encoding="utf-8"))
    split = load_case_split(args.split)

    from eval.predictors import CurlMeshGraphNetPredictor

    predictor = CurlMeshGraphNetPredictor.from_checkpoint(args.ckpt)
    axial = args.axial_length_m

    def band_of(rec):
        m = rec.mesh
        return build_airgap_band(m.node_x_mm, m.node_y_mm, m.tri, m.reg_code,
                                 m.name_of_code, moving_reg_codes=m.moving_reg_codes)

    print("=== 1. time_s sensitivity (case 4, step 0) ===")
    rec = load_doe_cases(manifest, args.data_dir, case_indices=[4]).records[0]
    band = band_of(rec)
    s0 = rec.samples[0]
    base = torque_of(rec, band, predictor, s0, axial)
    fem0 = arkkio_torque(band, s0.fields["bx"], s0.fields["by"], axial)
    print(f"  FEM {fem0:8.2f}   pred(t=0) {base:8.2f}   error {base - fem0:+8.2f}")
    for t in (rec.samples[1].time_s, rec.samples[10].time_s, 1e-3):
        patched = replace(s0, time_s=float(t))
        got = torque_of(rec, band, predictor, patched, axial)
        print(f"  time_s -> {t:10.3e}: pred {got:8.2f}  (moved {got - base:+7.2f} N*m)")

    print()
    print("=== 2. angle-encoding aliasing: step 0 vs last step ===")
    print("  (rotor_angle_features has a 2-sector period, so cum=0 and cum=-88 look alike)")
    from phase1_static.sector_symmetry import cumulative_rotor_angle, rotor_angle_features

    for case in list(split.test):
        rec = load_doe_cases(manifest, args.data_dir, case_indices=[case]).records[0]
        band = band_of(rec)
        cum = cumulative_rotor_angle([s.rotate_step for s in rec.samples])
        f_first, f_last = rotor_angle_features(cum[0]), rotor_angle_features(cum[-1])
        errs = []
        for idx in (0, len(rec.samples) - 1):
            s = rec.samples[idx]
            fem = arkkio_torque(band, s.fields["bx"], s.fields["by"], axial)
            errs.append(torque_of(rec, band, predictor, s, axial) - fem)
        print(f"  case {case:2d}: angle feat first {f_first[0]:+.3f},{f_first[1]:+.3f}  "
              f"last {f_last[0]:+.3f},{f_last[1]:+.3f}  |  err first {errs[0]:+7.1f}  last {errs[1]:+7.1f}")

    print()
    print("=== 3. train-set fit at early steps (underfit check) ===")
    for subset, cases in (("train", list(split.train)[:6]), ("test", list(split.test))):
        first, rest = [], []
        for case in cases:
            rec = load_doe_cases(manifest, args.data_dir, case_indices=[case]).records[0]
            band = band_of(rec)
            fem = np.array([arkkio_torque(band, s.fields["bx"], s.fields["by"], axial)
                            for s in rec.samples])
            sur = np.array([torque_of(rec, band, predictor, s, axial) for s in rec.samples])
            e = np.abs(sur - fem)
            first.append(e[:5].mean())
            rest.append(e[5:].mean())
        rel = 100 * np.mean(first) / 370.0
        print(f"  {subset:5s}: first5 |err| {np.mean(first):6.2f} N*m ({rel:.1f}% of rated)  "
              f"rest {np.mean(rest):6.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
