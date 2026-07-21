#!/usr/bin/env python3
"""Pull Motor-CAD's own on-load torque waveform for a DOE case.

Our benchmark's "FEM" torque is the harness integrating the exported element
field with Arkkio's method. That is not Motor-CAD's number, and the export
carries no torque of its own — the .mot holds only Lab (torque-speed envelope)
results at a different operating point, and the transient torque lives in the
binary .mes. So the absolute calibration of our torque metric has never been
checked against the solver that produced the fields.

This script closes that gap: it loads a case into Motor-CAD, tries to read the
cached solution's torque graph first, and only re-solves if the cache is empty.

It never touches the original solve. Point `--mot` at a copy.

Run with the Ansys Python environment that has pymotorcad, e.g.
    C:/Users/<user>/.ansys_python_venvs/PyMotorEnv_310/Scripts/python.exe \
        tools/fetch_motorcad_torque.py --mot <copy>/TestCAD1.mot --out results/motorcad_torque_case0004.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Motor-CAD graph names differ across versions and calculation types; try the
# documented ones in order and report which answered.
TORQUE_GRAPH_CANDIDATES = [
    "TorqueVW",
    "Torque (Virtual Work)",
    "TorqueMST",
    "Torque (Maxwell Stress)",
    "Torque",
    "Shaft Torque",
    "TorqueValues",
    "Cogging Torque",
    "CoggingTorqueVW",
]


def try_graphs(mc, names):
    """Return {name: (x, y)} for every graph name that answers."""
    found = {}
    for name in names:
        try:
            x, y = mc.get_magnetic_graph(name)
        except Exception as exc:  # graph absent or not calculated
            found[name] = ("ERR", str(exc)[:110])
            continue
        if x and y:
            found[name] = (list(x), list(y))
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mot", type=Path, required=True, help="Path to a COPY of the .mot")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--solve", action="store_true",
                        help="Run the E-Magnetic calculation if the cached graphs are empty")
    args = parser.parse_args()

    import ansys.motorcad.core as pymotorcad

    mc = pymotorcad.MotorCAD(open_new_instance=True)
    result = {"mot": str(args.mot), "solved": False}
    try:
        mc.load_from_file(str(args.mot.resolve()))
        print(f"loaded {args.mot}")

        for key in ("PeakCurrent", "PhaseAdvance", "Stator_Lam_Length",
                    "MagneticSymmetryFactor", "Pole_Number", "TorquePointsPerCycle"):
            try:
                result[key] = mc.get_variable(key)
            except Exception as exc:
                result[key] = f"ERR {exc}"
        print("  operating point:", {k: result[k] for k in ("PeakCurrent", "PhaseAdvance")})

        graphs = try_graphs(mc, TORQUE_GRAPH_CANDIDATES)
        good = {k: v for k, v in graphs.items() if v[0] != "ERR"}
        print(f"  cached graphs answering: {list(good)}")

        if not good and args.solve:
            print("  cache empty -> running E-Magnetic calculation (this takes minutes)")
            mc.do_magnetic_calculation()
            result["solved"] = True
            graphs = try_graphs(mc, TORQUE_GRAPH_CANDIDATES)
            good = {k: v for k, v in graphs.items() if v[0] != "ERR"}
            print(f"  graphs after solve: {list(good)}")

        result["graphs"] = {k: {"x": v[0], "y": v[1]} for k, v in good.items()}
        result["errors"] = {k: v[1] for k, v in graphs.items() if v[0] == "ERR"}

        for name, g in result["graphs"].items():
            y = g["y"]
            print(f"  {name:28s} n={len(y):3d}  mean {sum(y) / len(y):9.2f}  "
                  f"min {min(y):9.2f}  max {max(y):9.2f}")
    finally:
        try:
            mc.quit()
        except Exception:
            pass

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
