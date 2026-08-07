"""Verdict for the 120-step experiment: is the 45-point 21f a folded 24f?

Loads the 120-point re-solve of case 0007 (identical geometry and excitation, only
TorquePointsPerCycle changed) next to the 45-point original, computes the Arkkio
torque waveform and one-sided amplitude spectrum for both, and rules on:

  R1  ALIAS   -- amp(24f)@120 large while amp(21f)@120 small, with amp(24f)@120
                 close to the amp(21f) the 45-point grid reported -> the 45-point
                 "21f" was 24f folded across Nyquist 22.5.
  R2  GENUINE -- amp(21f)@120 close to amp(21f)@45 and amp(24f)@120 small.
  R3  MIXED   -- both present at 120; the 45-point bin held their unresolvable sum.

Also reports how the resolved orders (6/12/18) agree between grids -- they must match
within a few N*m/m or the two solves are not comparable and no verdict is valid.

    python results/logs/torque120_compare.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/workspace/app")

from eval.doe_dataset import load_doe_cases                    # noqa: E402
from eval.torque import arkkio_torque, build_airgap_band       # noqa: E402

EXP_DIR = Path("/workspace/exp")            # mount of D:/KDH/Sim_4SolverX/_torque120_case7
V2_DIR = Path("backup/doe_data_v2")


def torque_spectrum(rec):
    band = build_airgap_band(
        rec.mesh.node_x_mm, rec.mesh.node_y_mm, rec.mesh.tri,
        rec.mesh.reg_code, rec.mesh.name_of_code,
        moving_reg_codes=rec.mesh.moving_reg_codes,
    )
    tq = np.array([
        arkkio_torque(band,
                      np.asarray(s.fields["bx"], dtype=np.float64),
                      np.asarray(s.fields["by"], dtype=np.float64))
        for s in rec.samples
    ])
    n = tq.size
    amp = 2.0 * np.abs(np.fft.rfft(tq - tq.mean())) / n
    return tq, amp


def main() -> int:
    man120 = json.loads((EXP_DIR / "experiment_manifest.json").read_text(encoding="utf-8"))
    rec120 = load_doe_cases(man120, EXP_DIR, case_indices=[0]).records[0]
    man45 = json.loads((V2_DIR / "doe_manifest.json").read_text(encoding="utf-8"))
    rec45 = load_doe_cases(man45, V2_DIR, case_indices=[7]).records[0]

    tq120, amp120 = torque_spectrum(rec120)
    tq45, amp45 = torque_spectrum(rec45)
    print(f"120-step: {tq120.size} samples, mean {tq120.mean():9.2f} N*m/m, "
          f"p2p {tq120.max()-tq120.min():7.2f}")
    print(f" 45-step: {tq45.size} samples, mean {tq45.mean():9.2f} N*m/m, "
          f"p2p {tq45.max()-tq45.min():7.2f}")

    print(f"\n{'order':>6} {'amp@45':>9} {'amp@120':>9}")
    for m in (2, 4, 6, 9, 12, 18, 21, 24, 30, 36, 48):
        a45 = float(amp45[m]) if m < amp45.size else float("nan")
        a120 = float(amp120[m]) if m < amp120.size else float("nan")
        print(f"{m:>5}f {a45:>9.2f} {a120:>9.2f}")

    # comparability gate on the orders both grids resolve cleanly
    resolved_ok = all(
        abs(float(amp45[m]) - float(amp120[m])) < max(5.0, 0.15 * float(amp120[m]))
        for m in (6, 12)
    )
    a21_45 = float(amp45[21])
    a21_120, a24_120 = float(amp120[21]), float(amp120[24])

    if not resolved_ok:
        verdict = "INCOMPARABLE -- 6f/12f disagree between grids; investigate before ruling"
    elif a24_120 > 2.0 * a21_120 and a24_120 > 0.5 * a21_45:
        verdict = ("R1 ALIAS -- the 45-point 21f was a folded 24f "
                   f"(24f@120 {a24_120:.2f} vs 21f@120 {a21_120:.2f}; 45-grid showed {a21_45:.2f} at 21f)")
    elif a21_120 > 2.0 * a24_120:
        verdict = f"R2 GENUINE 21f ({a21_120:.2f}) -- negligible 24f ({a24_120:.2f})"
    else:
        verdict = f"R3 MIXED -- 21f {a21_120:.2f} and 24f {a24_120:.2f} both present"

    print(f"\nVERDICT: {verdict}")

    out = {
        "case": 7,
        "excitation": "byte-identical source .mot, RMS 460 A (section 26)",
        "n_steps": {"grid45": int(tq45.size), "grid120": int(tq120.size)},
        "mean_torque": {"grid45": float(tq45.mean()), "grid120": float(tq120.mean())},
        "p2p": {"grid45": float(tq45.max() - tq45.min()),
                "grid120": float(tq120.max() - tq120.min())},
        "amps_by_order": {str(m): {"grid45": (float(amp45[m]) if m < amp45.size else None),
                                   "grid120": (float(amp120[m]) if m < amp120.size else None)}
                          for m in (2, 4, 6, 9, 12, 18, 21, 24, 30, 36, 48)},
        "verdict": verdict,
    }
    Path("results/logs/torque120_compare_out.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    print("wrote results/logs/torque120_compare_out.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
