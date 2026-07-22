#!/usr/bin/env python3
"""Overlay FEM-Arkkio torque against N surrogate checkpoints on one axis.

Unlike tools/compare_torque_waveforms.py (single ckpt + Motor-CAD), this puts
several model versions head-to-head against the FEM ground truth for one case, so
a capacity/architecture change reads as a waveform, not just a scalar. The
per-model legend nRMSE is this one case's Arkkio-series nRMSE vs FEM — a viz
number, NOT the benchmark's test-set aggregate; they will differ.

    python tools/compare_model_torque.py --case 4 \
        --model "baseline h128=results/mgn_nodeB_notime_ep55_HPC134.pt" \
        --model "exp1 h256=results/mgn_nodeB_h256.pt" \
        --out results/viz/torque_cmp_case4.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root

import numpy as np  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from eval.doe_dataset import load_doe_cases  # noqa: E402
from eval.torque import arkkio_torque, build_airgap_band  # noqa: E402


def arkkio_series(record, band, axial_length_m, predictor=None):
    out = []
    for sample in record.samples:
        if predictor is None:
            bx, by = sample.fields["bx"], sample.fields["by"]
        else:
            pred = predictor.predict(record, sample)
            bx, by = pred[:, 0], pred[:, 1]
        out.append(arkkio_torque(band, bx, by, axial_length_m))
    return np.array(out)


def nrmse_pct(model, fem):
    return 100.0 * float(np.sqrt(np.mean((model - fem) ** 2)) / np.sqrt(np.mean(fem ** 2)))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data-dir", type=Path, default=Path("backup/doe_data"))
    p.add_argument("--case", type=int, required=True)
    p.add_argument("--model", action="append", required=True, metavar="LABEL=CKPT",
                   help="repeatable; e.g. 'exp1 h256=results/mgn_nodeB_h256.pt'")
    p.add_argument("--axial-length-m", type=float, default=0.150)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--dpi", type=int, default=200)
    args = p.parse_args()

    from eval.predictors import CurlMeshGraphNetPredictor  # torch, deferred

    manifest = json.loads((args.data_dir / "doe_manifest.json").read_text())
    report = load_doe_cases(manifest, args.data_dir, case_indices=[args.case])
    record = report.records[0]
    mesh = record.mesh
    band = build_airgap_band(
        mesh.node_x_mm, mesh.node_y_mm, mesh.tri, mesh.reg_code,
        mesh.name_of_code, moving_reg_codes=mesh.moving_reg_codes,
    )

    fem = arkkio_series(record, band, args.axial_length_m)
    n = len(fem)
    elec = np.arange(n) * (360.0 / n)  # electrical degrees over the sweep

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(elec, fem, "-", color="black", lw=2.2, label="FEM (ground truth)", zorder=3)

    styles = ["--", "-.", ":"]
    colors = ["tab:red", "tab:blue", "tab:green", "tab:purple"]
    for i, spec in enumerate(args.model):
        label, ckpt = spec.split("=", 1)
        pred = CurlMeshGraphNetPredictor.from_checkpoint(Path(ckpt), name=label.strip())
        series = arkkio_series(record, band, args.axial_length_m, predictor=pred)
        err = nrmse_pct(series, fem)
        ax.plot(elec, series, styles[i % len(styles)], color=colors[i % len(colors)],
                lw=1.8, label=f"{label.strip()}  (nRMSE {err:.1f}%)", zorder=2)

    ax.set_xlabel("electrical angle [deg]")
    ax.set_ylabel(f"torque [N.m]  (stack {args.axial_length_m*1000:.0f} mm)")
    ax.set_title(f"DOE case {args.case:04d} — torque waveform: FEM vs surrogates")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, facecolor="white")
    print(f"wrote {args.out}")
    print(f"  case {args.case}: {n} steps, FEM mean {fem.mean():.1f} N.m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
