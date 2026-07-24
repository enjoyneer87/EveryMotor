#!/usr/bin/env python3
"""Post-hoc spectral projection of the predicted airgap-band field.

Diagnosis (methodology review §13-14): on low-ripple test cases the model's band
ripple is mostly noise — correlation 0.14-0.47 against FEM with 1.5-1.7x the
amplitude — and torque is the band integral of B_r*B_theta, so that noise feeds
straight into the torque waveform. The true band field is spectrally sparse
(anti-periodic 1/8 sector: odd multiples of 4 cycles/rev), while the noise is
broadband. Projecting the *predicted* band field onto the admissible spatial
orders strips the broadband part at zero training cost.

Honesty constraints built in:
  * Admissible orders are selected from TRAIN-case FEM fields only — the test
    set never influences the basis.
  * Basis adequacy is proven on FEM itself: projecting FEM through the chosen
    basis must leave FEM torque essentially unchanged (reported as
    fem_distortion_pct). If that number is not tiny, the basis is wrong and the
    test numbers are meaningless.
  * The projection touches ONLY the band elements fed to the Arkkio integral;
    the field metric elsewhere is untouched by construction.

    python tools/band_spectral_projection.py \
        --ckpt "h128=results/mgn_nodeB_notime_ep55_HPC134.pt" \
        --ckpt "h256=results/mgn_nodeB_h256.pt" \
        --out results/band_projection_eval.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root

import numpy as np  # noqa: E402

from eval.doe_dataset import load_doe_cases  # noqa: E402
from eval.torque import arkkio_torque, build_airgap_band  # noqa: E402

TRAIN_SELECT_CASES = [0, 1, 2]     # FEM-only order selection; never the test split
SELECT_STEP_STRIDE = 5
# Candidates are restricted to what the sector symmetry admits. The band spans a
# 45-deg arc: integer orders closer than 8 cycles/rev differ by less than one
# full cycle over the arc and are nearly collinear there, so an unrestricted
# integer sweep lets LSQ smear energy across degenerate neighbours (a first run
# picked {88,90,93,94} and destroyed 99% of FEM torque — caught by the
# distortion gate below). An anti-periodic field over a 45-deg sector contains
# ONLY odd multiples of 4 cycles/rev; those are spaced 8 apart and separate
# cleanly on the arc. Slotting shows up as sidebands (48±4(2m+1) -> 44, 52, ...)
# which are themselves odd multiples of 4, so the set is closed.
CANDIDATE_ORDERS = [4 * (2 * m + 1) for m in range(24)]   # 4, 12, ..., 188
# Measured: with every admissible order up to 124 the FEM torque distortion
# plateaus at 0.234% — the element-wise (staircase) band field is not perfectly
# band-limited, and that residual is a discretization floor, not missing
# physics. The gate sits above that floor; it still rejects a wrong basis by
# two orders of magnitude (the degenerate integer sweep scored 99% distortion),
# while 0.3% is ~20x smaller than the multi-percent model effects under test.
FEM_DISTORTION_LIMIT_PCT = 0.3


def band_angles(band) -> np.ndarray:
    return np.arctan2(band.sin_theta, band.cos_theta)


def design_matrix(phi: np.ndarray, orders) -> np.ndarray:
    cols = []
    for k in orders:
        if k == 0:
            cols.append(np.ones_like(phi))
        else:
            cols.append(np.cos(k * phi))
            cols.append(np.sin(k * phi))
    return np.column_stack(cols)


def fit_energy_per_order(phi, values, orders):
    """LSQ once over the full candidate set; energy a_k^2+b_k^2 per order."""
    A = design_matrix(phi, orders)
    coef, *_ = np.linalg.lstsq(A, values, rcond=None)
    energy, i = {}, 0
    for k in orders:
        if k == 0:
            energy[k] = float(coef[i] ** 2)
            i += 1
        else:
            energy[k] = float(coef[i] ** 2 + coef[i + 1] ** 2)
            i += 2
    return energy


def projector(phi: np.ndarray, orders) -> np.ndarray:
    """P = A (A^T A)^-1 A^T for the restricted basis (band is stationary, reuse per case)."""
    A = design_matrix(phi, orders)
    return A @ np.linalg.pinv(A)


def band_rtheta(band, bx_full, by_full):
    bx, by = bx_full[band.element_index], by_full[band.element_index]
    br = bx * band.cos_theta + by * band.sin_theta
    bt = -bx * band.sin_theta + by * band.cos_theta
    return br, bt


def project_band(band, P, bx_full, by_full):
    """Return copies of the full element field with the band slice projected."""
    br, bt = band_rtheta(band, bx_full, by_full)
    br_p, bt_p = P @ br, P @ bt
    bx_p = br_p * band.cos_theta - bt_p * band.sin_theta
    by_p = br_p * band.sin_theta + bt_p * band.cos_theta
    bx2, by2 = bx_full.copy(), by_full.copy()
    bx2[band.element_index] = bx_p
    by2[band.element_index] = by_p
    return bx2, by2


def torque_series(rec, band, axial, predictor=None, P=None):
    out = []
    for s in rec.samples:
        if predictor is None:
            bx, by = s.fields["bx"], s.fields["by"]
        else:
            p = predictor.predict(rec, s)
            bx, by = p[:, 0], p[:, 1]
        if P is not None:
            bx, by = project_band(band, P, np.asarray(bx, float), np.asarray(by, float))
        out.append(arkkio_torque(band, bx, by, axial))
    return np.array(out)


def nrmse(pred, true):
    return 100.0 * float(np.sqrt(np.mean((pred - true) ** 2)) / np.sqrt(np.mean(true ** 2)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", type=Path, default=Path("backup/doe_data"))
    ap.add_argument("--ckpt", action="append", required=True, metavar="LABEL=PATH")
    ap.add_argument("--axial-length-m", type=float, default=0.150)
    ap.add_argument("--test-cases", type=int, nargs="+", default=[4, 7, 18, 32, 37, 39])
    ap.add_argument("--out", type=Path, default=Path("results/band_projection_eval.json"))
    args = ap.parse_args()

    man = json.loads((args.data_dir / "doe_manifest.json").read_text())

    # ---- 1) order selection on TRAIN FEM only -------------------------------
    print("== selecting admissible orders from TRAIN FEM ==")
    energy_sum: dict = {}
    train_bands = {}
    for c in TRAIN_SELECT_CASES:
        rec = load_doe_cases(man, args.data_dir, case_indices=[c]).records[0]
        m = rec.mesh
        band = build_airgap_band(m.node_x_mm, m.node_y_mm, m.tri, m.reg_code,
                                 m.name_of_code, moving_reg_codes=m.moving_reg_codes)
        train_bands[c] = (rec, band)
        phi = band_angles(band)
        for s in rec.samples[::SELECT_STEP_STRIDE]:
            br, bt = band_rtheta(band, np.asarray(s.fields["bx"], float),
                                 np.asarray(s.fields["by"], float))
            for vals in (br, bt):
                for k, e in fit_energy_per_order(phi, vals, CANDIDATE_ORDERS).items():
                    energy_sum[k] = energy_sum.get(k, 0.0) + e
    ranked = sorted(energy_sum, key=energy_sum.get, reverse=True)
    total_e = sum(energy_sum.values())

    # ---- 2) basis adequacy: smallest N whose FEM torque distortion is tiny --
    chosen = None
    for n in (4, 6, 8, 10, 12, 16, 20, 24, 32):
        orders = ranked[:n]
        worst = 0.0
        for c, (rec, band) in train_bands.items():
            phi = band_angles(band)
            P = projector(phi, orders)
            t_raw = torque_series(rec, band, args.axial_length_m)
            t_prj = torque_series(rec, band, args.axial_length_m, P=P)
            worst = max(worst, nrmse(t_prj, t_raw))
        kept_e = 100.0 * sum(energy_sum[k] for k in orders) / total_e
        print(f"  N={n:>2}  orders={sorted(orders)}  FEM-energy kept {kept_e:.3f}%  "
              f"FEM torque distortion {worst:.4f}%")
        if worst < FEM_DISTORTION_LIMIT_PCT and chosen is None:
            chosen = list(orders)
    if chosen is None:
        print("!! no basis met the FEM-distortion limit; refusing to project")
        return 1
    print(f"== chosen basis: {sorted(chosen)} ==")

    # ---- 3) test evaluation --------------------------------------------------
    from eval.predictors import CurlMeshGraphNetPredictor  # torch, deferred

    models = {}
    for spec in args.ckpt:
        label, path = spec.split("=", 1)
        models[label.strip()] = CurlMeshGraphNetPredictor.from_checkpoint(
            Path(path), name=label.strip())

    result = {"chosen_orders": sorted(int(k) for k in chosen),
              "fem_distortion_limit_pct": FEM_DISTORTION_LIMIT_PCT,
              "train_select_cases": TRAIN_SELECT_CASES,
              "per_case": {}, "aggregate": {}}
    fem_all, series_all = {}, {lbl: {"raw": {}, "proj": {}} for lbl in models}

    print("\n== test evaluation ==")
    print(f"{'case':>5} {'model':>6} {'raw nRMSE%':>11} {'proj nRMSE%':>12} {'delta':>7}")
    for c in args.test_cases:
        rec = load_doe_cases(man, args.data_dir, case_indices=[c]).records[0]
        m = rec.mesh
        band = build_airgap_band(m.node_x_mm, m.node_y_mm, m.tri, m.reg_code,
                                 m.name_of_code, moving_reg_codes=m.moving_reg_codes)
        P = projector(band_angles(band), chosen)
        fem = torque_series(rec, band, args.axial_length_m)
        fem_all[c] = fem
        result["per_case"][c] = {}
        for lbl, pred in models.items():
            raw = torque_series(rec, band, args.axial_length_m, predictor=pred)
            prj = torque_series(rec, band, args.axial_length_m, predictor=pred, P=P)
            series_all[lbl]["raw"][c], series_all[lbl]["proj"][c] = raw, prj
            r, p = nrmse(raw, fem), nrmse(prj, fem)
            result["per_case"][c][lbl] = {"raw_nrmse_pct": r, "proj_nrmse_pct": p}
            print(f"{c:>5} {lbl:>6} {r:>11.2f} {p:>12.2f} {p - r:>+7.2f}")

    F = np.concatenate([fem_all[c] for c in args.test_cases])
    for lbl in models:
        R = np.concatenate([series_all[lbl]["raw"][c] for c in args.test_cases])
        Pj = np.concatenate([series_all[lbl]["proj"][c] for c in args.test_cases])
        result["aggregate"][lbl] = {"raw_nrmse_pct": nrmse(R, F),
                                    "proj_nrmse_pct": nrmse(Pj, F)}
        print(f"\n  aggregate {lbl}: raw {nrmse(R, F):.3f}%  ->  projected {nrmse(Pj, F):.3f}%")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
