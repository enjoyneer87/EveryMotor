"""Prior-level excitation gate: does current_scale actually reach the prior field?

The section-24 defect was a DOE axis that was written but never consumed. The same
failure can hide one layer down: if fem_warmstart/prior.py computes its linear solve at
the template current regardless of the case's PeakCurrent, then every R8 case at 325.3 A
or 162.6 A gets a prior drawn at 650.5 A, and the network's "physics prior" is blind to
the very axis R8 exists to add.

Two checks, both on the linear prior so they are exact rather than statistical:

1. REACHES   -- B(scale=0) != B(scale=1). Magnets alone vs magnets + full current.
                If these agree, the current is not wired into the prior at all.
2. SUPERPOSES -- B(0.5) == 0.5 * (B(0) + B(1)) to solver precision. The prior is a
                linear solve, so the current-driven part must scale exactly. This
                catches a scale that is applied but applied wrongly (squared,
                offset, clipped).

Finally it reports the scale the manifest implies for a real R8 case, so a mis-anchored
TEMPLATE_IPK_A shows up as a number a human can check against section 26 (650.538 A).

    python results/logs/verify_prior_current.py --data-dir backup/doe_data_v2 --case 240
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/workspace/app")

from eval.doe_dataset import load_doe_cases                      # noqa: E402
from fem_warmstart.prior import (                                # noqa: E402
    TEMPLATE_IPK_A, case_current_scale, compute_prior,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", type=Path, default=Path("backup/doe_data_v2"))
    ap.add_argument("--case", type=int, default=240)
    ap.add_argument("--step", type=int, default=0)
    ap.add_argument("--tol", type=float, default=1e-9)
    args = ap.parse_args()

    import json
    manifest = json.loads((args.data_dir / "doe_manifest.json").read_text(encoding="utf-8"))
    rec = load_doe_cases(manifest, args.data_dir, case_indices=[args.case]).records[0]

    ipk = float(rec.condition.get("PeakCurrent", float("nan")))
    implied = case_current_scale(rec)
    print(f"case {args.case}: PeakCurrent {ipk:.3f} A / template {TEMPLATE_IPK_A:.3f} A "
          f"-> current_scale {implied:.6f}  (wired={rec.condition.get('excitation_wired')})")

    b0 = compute_prior(rec, args.step, current_scale=0.0)[:, 1:]
    b1 = compute_prior(rec, args.step, current_scale=1.0)[:, 1:]
    bh = compute_prior(rec, args.step, current_scale=0.5)[:, 1:]

    ref = float(np.abs(b1).max())
    reach = float(np.abs(b1 - b0).max()) / max(ref, 1e-30)
    lin = float(np.abs(bh - 0.5 * (b0 + b1)).max()) / max(ref, 1e-30)

    print(f"  REACHES   max|B(1)-B(0)| / max|B(1)| = {reach:.6e}   (must be >> 0)")
    print(f"  SUPERPOSES max|B(.5)-mean| / max|B(1)| = {lin:.6e}   (must be ~0)")

    ok = reach > 1e-3 and lin < max(args.tol, 1e-9)
    print("PRIOR_CURRENT_GATE_" + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
