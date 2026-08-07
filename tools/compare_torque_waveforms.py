#!/usr/bin/env python3
"""Overlay Motor-CAD, FEM-Arkkio and surrogate-Arkkio torque waveforms.

Three curves, one axis, one sign convention:

1. **Motor-CAD TorqueVW** — the solver's own virtual-work torque, from a
   re-run of the E-Magnetic calculation (`tools/fetch_motorcad_torque.py`).
2. **Arkkio on the FEM field** — the harness integrating the exported element B.
   This is what the benchmark calls "FEM".
3. **Arkkio on the surrogate field** — the same integral applied to the model's
   predicted B, so any difference is the model's, not the operator's.

Sign convention
---------------
`eval.torque.arkkio_torque` returns torque about **+theta** (counter-clockwise).
In this export the rotor turns clockwise (`rotate_step < 0`, rotor mean angle
falls from -43 to -120 deg over the sweep), and the machine is motoring, so the
electromagnetic torque points along the motion — clockwise — and our integral is
correctly **negative**. Motor-CAD reports motoring torque as a positive
magnitude. The two differ by convention, not physics.

This script therefore plots `-T` for our curves and says so on the figure.
The flip cannot be hiding a real sign error: the waveform's mean (~368 N·m)
dwarfs its ripple (+-28 N·m), so it never crosses zero and **no phase shift can
map +370 onto -370** — fitting Motor-CAD against our unflipped curve gives an
RMSE 370x worse than against the flipped one.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from eval.doe_dataset import load_doe_cases  # noqa: E402
from eval.torque import arkkio_torque, build_airgap_band  # noqa: E402


def arkkio_series(record, band, axial_length_m, predictor=None):
    """Torque per step from the FEM field, or from a predictor's field."""
    out = []
    for sample in record.samples:
        if predictor is None:
            bx, by = sample.fields["bx"], sample.fields["by"]
        else:
            pred = predictor.predict(record, sample)
            bx, by = pred[:, 0], pred[:, 1]
        out.append(arkkio_torque(band, bx, by, axial_length_m))
    return np.array(out)


def best_phase_shift(elec_deg, target, mc_x, mc_y):
    """Electrical-degree shift aligning the Motor-CAD curve to ours."""
    best = (0, float("inf"))
    for shift in range(360):
        curve = np.interp((elec_deg + shift) % 360, mc_x, mc_y, period=360)
        rmse = float(np.sqrt(np.mean((curve - target) ** 2)))
        if rmse < best[1]:
            best = (shift, rmse)
    return best


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=Path("backup/doe_data"))
    parser.add_argument("--case", type=int, default=4)
    parser.add_argument("--ckpt", type=Path, default=Path("results/mgn_nodeB_long.pt"))
    parser.add_argument("--motorcad", type=Path,
                        default=Path("results/motorcad_torque_case0004.json"))
    parser.add_argument("--out", type=Path,
                        default=Path("results/viz/torque_three_way_case0004.png"))
    args = parser.parse_args()

    mc = json.loads(args.motorcad.read_text(encoding="utf-8"))
    mc_x = np.array(mc["graphs"]["TorqueVW"]["x"])
    mc_y = np.array(mc["graphs"]["TorqueVW"]["y"])
    axial = mc["Stator_Lam_Length"] / 1000.0
    pole_pairs = mc["Pole_Number"] // 2

    manifest = json.loads((args.data_dir / "doe_manifest.json").read_text(encoding="utf-8"))
    record = load_doe_cases(manifest, args.data_dir, case_indices=[args.case]).records[0]
    mesh = record.mesh
    band = build_airgap_band(
        mesh.node_x_mm, mesh.node_y_mm, mesh.tri, mesh.reg_code,
        mesh.name_of_code, moving_reg_codes=mesh.moving_reg_codes,
    )

    from eval.predictors import CurlMeshGraphNetPredictor

    predictor = CurlMeshGraphNetPredictor.from_checkpoint(args.ckpt)

    t_fem = arkkio_series(record, band, axial)
    t_sur = arkkio_series(record, band, axial, predictor)

    rot = np.cumsum([s.rotate_step for s in record.samples])
    rot -= rot[0]
    elec = np.abs(rot) * pole_pairs

    # Motor-CAD's positive-motoring convention; see the module docstring.
    fem, sur = -t_fem, -t_sur
    shift, rmse = best_phase_shift(elec, fem, mc_x, mc_y)
    mc_aligned = np.interp((elec + shift) % 360, mc_x, mc_y, period=360)

    def stats(name, a, ref):
        print(f"  {name:26s} mean {a.mean():8.2f}  ripple {a.max() - a.min():7.2f}  "
              f"vs ref mean {100 * (a.mean() - ref.mean()) / ref.mean():+6.2f}%  "
              f"RMSE {np.sqrt(np.mean((a - ref) ** 2)):6.2f}")

    print(f"case {args.case}: {len(elec)} steps, elec sweep {elec.min():.0f}-{elec.max():.0f} deg, "
          f"stack {axial * 1e3:.0f} mm, phase shift {shift} deg (RMSE {rmse:.2f})")
    stats("Motor-CAD TorqueVW", mc_aligned, mc_aligned)
    stats("Arkkio on FEM B", fem, mc_aligned)
    stats("Arkkio on surrogate B", sur, mc_aligned)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Display-only band-limited interpolation (section 28): the 45 samples span one
    # electrical cycle, and every order below ~18f is faithful, so the smooth curve is
    # the waveform itself rather than a spline. Straight lines between samples are what
    # made this figure look ragged — the dominant ripple is 12f at 3.75 samples/period.
    # All three curves are upsampled from the SAME uniform 45-point grid so they stay
    # comparable; markers keep the actual sample positions visible.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from make_field_gif import fourier_upsample

    spacing = float(elec[1] - elec[0])
    xd, mc_d = fourier_upsample(mc_aligned)
    _, fem_d = fourier_upsample(fem)
    _, sur_d = fourier_upsample(sur)
    elec_d = elec[0] + xd * spacing

    fig, (ax, axe) = plt.subplots(2, 1, figsize=(10, 6.6), height_ratios=[2.3, 1], sharex=True)
    ax.plot(elec_d, mc_d, "-", color="#1f77b4", lw=2.0,
            label="Motor-CAD  TorqueVW (virtual work)")
    ax.plot(elec_d, fem_d, "--", color="#2ca02c", lw=1.7,
            label="Arkkio on FEM B  (harness \"FEM\")")
    ax.plot(elec_d, sur_d, "-.", color="#d62728", lw=1.7,
            label="Arkkio on surrogate B  (PhysicsNeMo MGN)")
    ax.plot(elec, mc_aligned, "o", color="#1f77b4", ms=3.2, alpha=0.45)
    ax.plot(elec, fem, "s", color="#2ca02c", ms=3.2, alpha=0.45)
    ax.plot(elec, sur, "^", color="#d62728", ms=3.2, alpha=0.45)
    ax.set_ylabel("torque  [N·m]")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="lower right")
    ax.set_title(
        f"DOE case_{args.case:04d} — torque waveform, three ways\n"
        f"{mc['PeakCurrent']:.1f} A, phase adv {mc['PhaseAdvance']:.2f}°, 8-pole 1/8 sector, "
        f"stack {axial * 1e3:.0f} mm   |   our curves negated to Motor-CAD's "
        f"positive-motoring convention   |   smooth curves: band-limited interpolation "
        f"of the 45-step grid (§28)",
        fontsize=9.5,
    )

    axe.plot(elec_d, fem_d - mc_d, color="#2ca02c", lw=1.6, label="FEM − Motor-CAD")
    axe.plot(elec_d, sur_d - mc_d, color="#d62728", lw=1.6, label="surrogate − Motor-CAD")
    axe.axhline(0, color="0.6", lw=0.8)
    axe.set_ylabel("error  [N·m]")
    axe.set_xlabel("rotor position  [electrical deg]")
    axe.grid(alpha=0.3)
    axe.legend(fontsize=8, ncol=2)

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130, facecolor="white")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
