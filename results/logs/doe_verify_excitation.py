"""Post-generation excitation gate: did the current axis actually reach the fields?

The section-24 failure mode — a DOE axis written to the .mot but never consumed
by the solve — is invisible to every downstream check (the solve succeeds, the
export parses, training converges). This gate makes it loud: for each sampled
case, fit the per-conductor ampere-turn amplitude from the exported j (the
section-24/A3 method) and compare with the amplitude the case's nominal
PeakCurrent implies (A-turns = I_pk / ParallelPaths = I_pk / 2 on this template).

Run after ingest, pointing at the ingested data dir:

    python results/logs/doe_verify_excitation.py --data-dir backup/doe_data_v2 --cases 0 5 11

Exits 1 unless every sampled case matches within --tol (default 2%).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "D:/KDH/NvidiaNemo")

from eval.doe_dataset import load_doe_cases                 # noqa: E402
from eval.mesh_regions import element_areas_m2              # noqa: E402
from fem_warmstart.winding import ELEC_DEG_PER_STEP, phase_group  # noqa: E402

PARALLEL_PATHS = 2.0


def measured_ampere_turns(rec) -> float:
    """Median per-conductor ampere-turn amplitude across conductor regions."""
    mesh = rec.mesh
    reg = np.asarray(mesh.reg_code)
    steps = (0, 5, 11, 17, 23, 29, 35, 41)
    omega = np.radians(ELEC_DEG_PER_STEP) * np.asarray(steps, dtype=np.float64)
    design = np.column_stack([np.cos(omega), -np.sin(omega)])
    area = element_areas_m2(rec.samples[0].node_x_mm, rec.samples[0].node_y_mm, mesh.tri)
    amps = []
    for code, name in mesh.name_of_code.items():
        if phase_group(str(name)) is None:
            continue
        m = reg == int(code)
        jj = np.array([float(np.asarray(rec.samples[k].fields["j"])[m][0]) for k in steps])
        (c, s), *_ = np.linalg.lstsq(design, jj, rcond=None)
        amps.append(float(np.hypot(c, s)) * 1e6 * float(area[m].sum()))
    return float(np.median(amps))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--cases", type=int, nargs="+", default=[0])
    ap.add_argument("--tol", type=float, default=0.02)
    args = ap.parse_args()

    manifest = json.loads((args.data_dir / "doe_manifest.json").read_text(encoding="utf-8"))
    ok = True
    for ci in args.cases:
        rec = load_doe_cases(manifest, args.data_dir, case_indices=[ci]).records[0]
        ipk = float(rec.condition.get("PeakCurrent", float("nan")))
        expect = ipk / PARALLEL_PATHS
        got = measured_ampere_turns(rec)
        rel = abs(got - expect) / expect if expect > 0 else float("inf")
        status = "OK" if rel < args.tol else "WIRING DEAD"
        print(f"case {ci:4d}: nominal Ipk {ipk:8.2f} A -> expect {expect:7.2f} A-turns, "
              f"measured {got:7.2f}  ({100*rel:.2f}% off)  {status}")
        ok &= rel < args.tol
    print("EXCITATION_GATE_" + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
