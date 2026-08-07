"""Is 45 steps/electrical-cycle under-resolving the torque waveform?

The user observed (training-viz gif) that the torque trace looks ragged and does not
show the six ripple periods a 3-phase machine's 6f torque harmonic should produce.
Before generating 120-step data, quantify what the existing 45-step TRUTH actually
contains: compute the Arkkio torque of the ground-truth fields per rotor step, FFT it
over the electrical cycle, and report (a) harmonic amplitudes by electrical order,
(b) samples-per-period for each order at 45 vs 120 steps/cycle, (c) how much of the
ripple RMS sits in orders at or above the 45-step Nyquist (22.5f) -- energy there is
either invisible or ALIASED into lower orders, which would corrupt the ripple metric
for both truth and prediction alike.

    python results/logs/_torque_step_diag.py --data-dir backup/doe_data_v2 --cases 4 240 241
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/workspace/app")

from eval.doe_dataset import load_doe_cases                    # noqa: E402
from eval.torque import arkkio_torque, build_airgap_band       # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", type=Path, default=Path("backup/doe_data_v2"))
    ap.add_argument("--cases", type=int, nargs="+", default=[4, 240, 241])
    ap.add_argument("--out", type=Path,
                    default=Path("results/logs/_torque_step_diag_out.json"))
    args = ap.parse_args()

    manifest = json.loads((args.data_dir / "doe_manifest.json").read_text(encoding="utf-8"))
    report = {}
    for ci in args.cases:
        rec = load_doe_cases(manifest, args.data_dir, case_indices=[ci]).records[0]
        n_steps = len(rec.samples)
        # Same band construction as eval.benchmark: built once per record on the
        # stationary stator-side airgap layer, truth fields sliced per step.
        band = build_airgap_band(
            rec.mesh.node_x_mm, rec.mesh.node_y_mm, rec.mesh.tri,
            rec.mesh.reg_code, rec.mesh.name_of_code,
            moving_reg_codes=rec.mesh.moving_reg_codes,
        )
        tq = np.empty(n_steps)
        for k, s in enumerate(rec.samples):
            tq[k] = arkkio_torque(band,
                                  np.asarray(s.fields["bx"], dtype=np.float64),
                                  np.asarray(s.fields["by"], dtype=np.float64))

        mean = float(tq.mean())
        ripple = tq - mean
        # rFFT over the (assumed periodic) electrical cycle: bin m = electrical order m
        spec = np.fft.rfft(ripple) / n_steps
        amp = 2.0 * np.abs(spec)                      # one-sided amplitude per order
        orders = np.arange(amp.size)
        top = sorted(((float(amp[m]), int(m)) for m in orders[1:]), reverse=True)[:8]

        ripple_rms = float(np.sqrt((ripple ** 2).mean()))
        # energy in orders the 45-step grid cannot represent cleanly (>= Nyquist 22.5)
        hi = float(np.sqrt(np.sum((amp[orders >= 22] / np.sqrt(2)) ** 2)))
        p2p = float(tq.max() - tq.min())

        report[str(ci)] = {
            "n_steps": n_steps,
            "torque_mean": mean,
            "ripple_rms": ripple_rms,
            "ripple_p2p": p2p,
            "ripple_pct_of_mean": 100.0 * p2p / abs(mean) if mean else None,
            "top_orders_amp": [{"order": m, "amp": a,
                                "pts_per_period_45": 45.0 / m,
                                "pts_per_period_120": 120.0 / m} for a, m in top],
            "rms_at_or_above_order22": hi,
            "hi_order_share_pct": 100.0 * hi / ripple_rms if ripple_rms else None,
        }
        tops = ", ".join(f"{m}f:{a:.2f}" for a, m in top[:5])
        print(f"case {ci:4d} ({n_steps} steps): mean {mean:9.2f} N*m/m  "
              f"ripple p2p {p2p:7.2f} ({report[str(ci)]['ripple_pct_of_mean']:.1f}% of mean)  "
              f"top orders [{tops}]  >=22f share {report[str(ci)]['hi_order_share_pct']:.1f}%")

    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
