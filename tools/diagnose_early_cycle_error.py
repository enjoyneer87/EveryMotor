#!/usr/bin/env python3
"""Diagnose the surrogate's early-cycle torque error.

On case_0004 the surrogate's torque is up to 110 N*m low over the first ~100
electrical degrees and settles afterwards. Three explanations were on the table:

(a) **Feature artifact** — step 0 has ``rotate_step = 0`` where every other step
    has -2, and ``time_s`` is NaN in the export (the loader maps it to 0).
(b) **Data/geometry artifact** — something about the early steps in the export.
(c) **MGN limitation** — capacity or generalization, not the data.

The discriminating evidence is *where* the error lives:

* repeats at the same electrical angle in **every** case  -> systematic, (a)/(b)
* appears in **train** cases too                          -> not generalization
* varies per case with no common angle                    -> capacity, (c)

This script computes the per-step torque error for train and holdout cases and
reports those three cuts.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from eval.case_split import load_case_split  # noqa: E402
from eval.doe_dataset import load_doe_cases  # noqa: E402
from eval.torque import arkkio_torque, build_airgap_band  # noqa: E402


def per_step_error(record, predictor, axial_length_m):
    """Signed torque error per step, plus the FEM reference."""
    mesh = record.mesh
    band = build_airgap_band(
        mesh.node_x_mm, mesh.node_y_mm, mesh.tri, mesh.reg_code,
        mesh.name_of_code, moving_reg_codes=mesh.moving_reg_codes,
    )
    fem, sur = [], []
    for sample in record.samples:
        fem.append(arkkio_torque(band, sample.fields["bx"], sample.fields["by"], axial_length_m))
        pred = predictor.predict(record, sample)
        sur.append(arkkio_torque(band, pred[:, 0], pred[:, 1], axial_length_m))
    return np.array(fem), np.array(sur)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=Path("backup/doe_data"))
    parser.add_argument("--ckpt", type=Path, default=Path("results/mgn_nodeB_long.pt"))
    parser.add_argument("--split", type=Path, default=Path("eval/splits/doe40_case_split.json"))
    parser.add_argument("--n-train", type=int, default=6, help="How many train cases to include")
    parser.add_argument("--axial-length-m", type=float, default=0.150)
    parser.add_argument("--out", type=Path, default=Path("results/early_cycle_diagnosis.json"))
    args = parser.parse_args()

    manifest = json.loads((args.data_dir / "doe_manifest.json").read_text(encoding="utf-8"))
    split = load_case_split(args.split)
    train_cases = list(split.train)[: args.n_train]
    test_cases = list(split.test)

    from eval.predictors import CurlMeshGraphNetPredictor

    predictor = CurlMeshGraphNetPredictor.from_checkpoint(args.ckpt)

    out = {"train": {}, "test": {}, "axial_length_m": args.axial_length_m}
    for subset, cases in (("train", train_cases), ("test", test_cases)):
        for case in cases:
            record = load_doe_cases(manifest, args.data_dir, case_indices=[case]).records[0]
            fem, sur = per_step_error(record, predictor, args.axial_length_m)
            rot = np.cumsum([s.rotate_step for s in record.samples])
            rot -= rot[0]
            out[subset][str(case)] = {
                "elec_deg": (np.abs(rot) * 4).tolist(),
                "t_fem": fem.tolist(),
                "t_sur": sur.tolist(),
                "time_s": [s.time_s for s in record.samples],
                "rotate_step": [s.rotate_step for s in record.samples],
            }
            err = sur - fem
            print(f"  {subset:5s} case {case:2d}: |err| first5 {np.abs(err[:5]).mean():7.2f}  "
                  f"rest {np.abs(err[5:]).mean():7.2f}  ratio {np.abs(err[:5]).mean() / max(np.abs(err[5:]).mean(), 1e-9):5.2f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
