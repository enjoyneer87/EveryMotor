"""Decisive experiment: re-solve ONE case at 120 torque points per cycle.

The 45-step spectrum (results/logs/_torque_step_diag_out.json) shows a 21f component
whose provenance is ambiguous: 45 points put Nyquist at order 22.5, so a TRUE 24f slot
harmonic (2x the dominant 12f) would fold to |45-24| = 21f -- indistinguishable from a
genuine 21f in the 45-point data alone. At 120 points (Nyquist 60) the orders 21 and 24
are directly resolved, settling the question with one solve. Case 0007 is chosen
because it carries the largest 21f of the audited cases (26.9 N*m/m).

Section-8b protocol: the solve runs on a COPY of the original .mot in a scratch
campaign dir; DOE4TrainingData is never touched. The excitation is deliberately left
alone -- the original RMS-mode 460 A is exactly what the 45-step truth was solved at
(section 26), so the two spectra differ in nothing but the step count.

Run with the Ansys venv:
  C:/Users/moa/.ansys_python_venvs/PyMotorEnv_310/Scripts/python.exe results/logs/torque120_experiment.py
"""
import json
import pathlib
import shutil
import sys
import time

REPO = pathlib.Path(r"D:/KDH/NvidiaNemo")
sys.path.insert(0, str(REPO / "eMach"))

from tools.motorCAD.pyMCAD.doe_batch import (  # noqa: E402
    doe_export_txt,
    doe_export_h5_from_txt,
)
import ansys.motorcad.core as pmc  # noqa: E402

SRC_MOT = pathlib.Path(r"D:/KDH/Sim_4SolverX/DOE4TrainingData/case_0007/TestCAD1.mot")
OUT = pathlib.Path(r"D:/KDH/Sim_4SolverX/_torque120_case7")
POINTS = 120


def main() -> int:
    case_dir = OUT / "case_0000"
    case_dir.mkdir(parents=True, exist_ok=True)
    case_mot = case_dir / "TestCAD1.mot"
    shutil.copy2(SRC_MOT, case_mot)

    mc = pmc.MotorCAD()
    try:
        mc.load_from_file(str(case_mot))
        mc.set_variable("MessageDisplayState", 2)
        prev_pts = mc.get_variable("TorquePointsPerCycle")
        prev_def = mc.get_variable("CurrentDefinition")
        prev_rms = mc.get_variable("RMSCurrent")
        print(f"source state: TorquePointsPerCycle={prev_pts}  "
              f"CurrentDefinition={prev_def}  RMSCurrent={prev_rms}", flush=True)
        mc.set_variable("TorquePointsPerCycle", POINTS)
        mc.save_to_file(str(case_mot))
        try:
            mc.display_screen("E-Magnetics;FEA")
        except Exception:                                       # noqa: BLE001
            pass
        t0 = time.time()
        mc.do_magnetic_calculation()
        solve_s = time.time() - t0
        mc.save_to_file(str(case_mot))
        doe_export_txt(mc, case_dir, verbose=False)
        doe_export_h5_from_txt(case_dir, verbose=False)
    finally:
        try:
            mc.quit()
        except Exception:                                       # noqa: BLE001
            pass

    h5s = sorted((case_dir / "postproc").glob("*.h5"))
    (OUT / "experiment_manifest.json").write_text(json.dumps({
        "n_cases": 1,
        "campaign": "torque120_step_resolution_experiment",
        "note": (f"case_0007 geometry+excitation byte-identical to DOE4TrainingData; "
                 f"only TorquePointsPerCycle {int(prev_pts)} -> {POINTS}"),
        "cases": [{
            "index": 0,
            "source_geometry_index": 7,
            "geometry": {},
            "electrical": {"RMSCurrent": prev_rms, "CurrentDefinition": prev_def,
                           "TorquePointsPerCycle": POINTS},
            "h5_paths": [str(p) for p in h5s],
            "solve_time_s": solve_s,
        }],
    }, indent=1), encoding="utf-8")
    print(f"TORQUE120_SOLVE_DONE {len(h5s)} h5 in {solve_s:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
