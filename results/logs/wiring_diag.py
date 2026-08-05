"""Causal confirmation of the section-24 excitation-wiring root cause.

Evidence so far (textual): every case .mot carries CurrentDefinition=1 (RMS
mode) and the template's RMSCurrent=460, while the DOE wrote PeakCurrent —
which in RMS mode is a derived display quantity, not a solve input.
460*sqrt(2)/2 paths = 325.27 A-turns = the section-24 constant.

This script proves causality with two solves on a scratchpad copy of the
original test case_0004 (baseline TorqueVW mean = 370.13 N*m, section 8b):

    solve A: set PeakCurrent = 100  -> expect torque UNCHANGED (~370)
    solve B: set RMSCurrent  = 230  -> expect torque to DROP substantially

Run with the Ansys venv:
  C:/Users/moa/.ansys_python_venvs/PyMotorEnv_310/Scripts/python.exe wiring_diag.py
"""
import json
import shutil
import time
from pathlib import Path

import ansys.motorcad.core as pymotorcad

SRC = Path(r"D:/KDH/Sim_4SolverX/DOE4TrainingData/case_0004")
WORK = Path(__file__).resolve().parent / "wiring_diag_case"
OUT = Path(__file__).resolve().parent / "wiring_diag_result.json"

GRAPHS = ("TorqueVW", "Torque")


def torque_mean(mc):
    for name in GRAPHS:
        try:
            x, y = mc.get_magnetic_graph(name)
            if y:
                return sum(y) / len(y), name, len(y)
        except Exception:
            continue
    return None, None, 0


def solve_with(mc, mot, sets):
    mc.load_from_file(str(mot))
    mc.set_variable("MessageDisplayState", 2)
    before = {k: mc.get_variable(k) for k in
              ("CurrentDefinition", "RMSCurrent", "PeakCurrent", "PhaseAdvance")}
    for k, v in sets.items():
        mc.set_variable(k, v)
    after = {k: mc.get_variable(k) for k in before}
    t0 = time.time()
    mc.do_magnetic_calculation()
    tq, gname, npts = torque_mean(mc)
    return {"sets": sets, "before": before, "after": after,
            "torque_mean": tq, "graph": gname, "n_points": npts,
            "solve_s": round(time.time() - t0, 1)}


def main():
    if WORK.exists():
        shutil.rmtree(WORK)
    shutil.copytree(SRC, WORK)
    mot = WORK / "TestCAD1.mot"

    mc = pymotorcad.MotorCAD()
    results = {"baseline_ref": {"torque_mean": 370.13, "source": "section 8b re-solve, unmodified settings"}}
    try:
        results["A_peakcurrent_100"] = solve_with(mc, mot, {"PeakCurrent": 100.0})
        print("A done:", results["A_peakcurrent_100"]["torque_mean"], flush=True)
        results["B_rmscurrent_230"] = solve_with(mc, mot, {"RMSCurrent": 230.0})
        print("B done:", results["B_rmscurrent_230"]["torque_mean"], flush=True)
    finally:
        try:
            mc.quit()
        except Exception:
            pass
    OUT.write_text(json.dumps(results, indent=1), encoding="utf-8")
    print("WIRING_DIAG_DONE")


if __name__ == "__main__":
    main()
