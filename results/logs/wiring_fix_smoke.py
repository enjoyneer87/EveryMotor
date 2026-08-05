"""Verify the wiring FIX semantics: CurrentDefinition=0 makes PeakCurrent live.

Two solves with exactly known expected torques (scratchpad copy, originals
untouched):

    fix A: CurrentDefinition=0, PeakCurrent=650.538  -> same physical current as
           the RMS-mode template (460 RMS), expect TorqueVW mean = 370.13
    fix B: CurrentDefinition=0, PeakCurrent=325.269  -> same as section 26's
           RMSCurrent=230 solve, expect TorqueVW mean = 210.16

Passing both proves mode 0 = peak-current definition AND that the recipe
reproduces known physics quantitatively.
"""
import json
import shutil
import time
from pathlib import Path

import ansys.motorcad.core as pymotorcad

SRC = Path(r"D:/KDH/Sim_4SolverX/DOE4TrainingData/case_0004")
WORK = Path(__file__).resolve().parent / "wiring_fix_case"
OUT = Path(__file__).resolve().parent / "wiring_fix_result.json"


def torque_mean(mc):
    x, y = mc.get_magnetic_graph("TorqueVW")
    return sum(y) / len(y)


def solve_with(mc, mot, sets):
    mc.load_from_file(str(mot))
    mc.set_variable("MessageDisplayState", 2)
    for k, v in sets.items():
        mc.set_variable(k, v)
    readback = {k: mc.get_variable(k) for k in
                ("CurrentDefinition", "RMSCurrent", "PeakCurrent")}
    t0 = time.time()
    mc.do_magnetic_calculation()
    return {"sets": sets, "readback_after_set": readback,
            "torque_mean": torque_mean(mc), "solve_s": round(time.time() - t0, 1)}


def main():
    if WORK.exists():
        shutil.rmtree(WORK)
    shutil.copytree(SRC, WORK)
    mot = WORK / "TestCAD1.mot"
    mc = pymotorcad.MotorCAD()
    res = {"expect_A": 370.13, "expect_B": 210.16}
    try:
        res["A_peak_650p538"] = solve_with(
            mc, mot, {"CurrentDefinition": 0, "PeakCurrent": 650.538238691624})
        print("A:", res["A_peak_650p538"]["torque_mean"], flush=True)
        res["B_peak_325p269"] = solve_with(
            mc, mot, {"CurrentDefinition": 0, "PeakCurrent": 325.269119345812})
        print("B:", res["B_peak_325p269"]["torque_mean"], flush=True)
    finally:
        try:
            mc.quit()
        except Exception:
            pass
    ok_a = abs(res["A_peak_650p538"]["torque_mean"] - 370.13) < 1.0
    ok_b = abs(res["B_peak_325p269"]["torque_mean"] - 210.16) < 1.0
    res["verdict"] = "FIX_VERIFIED" if (ok_a and ok_b) else "FIX_MISMATCH"
    OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(res["verdict"])


if __name__ == "__main__":
    main()
