"""Re-export the 120-point solve with the step cap lifted.

doe_export_txt defaults to final_step=45 (the campaign's TorquePointsPerCycle), so the
first export of the 120-point solve silently wrote steps 1..45 = 135 elec deg -- a
partial cycle whose FFT is meaningless (that is exactly what torque120_compare flagged:
45 samples and a leakage-scrambled spectrum). The .mes solution holds all 120 steps;
re-exporting is enough -- no re-solve.

  C:/Users/moa/.ansys_python_venvs/PyMotorEnv_310/Scripts/python.exe results/logs/torque120_reexport.py
"""
import pathlib
import sys
import time

REPO = pathlib.Path(r"D:/KDH/NvidiaNemo")
sys.path.insert(0, str(REPO / "eMach"))

from tools.motorCAD.pyMCAD.doe_batch import (  # noqa: E402
    doe_export_txt,
    doe_export_h5_from_txt,
)
import ansys.motorcad.core as pmc  # noqa: E402

CASE_DIR = pathlib.Path(r"D:/KDH/Sim_4SolverX/_torque120_case7/case_0000")
POINTS = 120


def main() -> int:
    mc = pmc.MotorCAD()
    try:
        mc.set_variable("MessageDisplayState", 2)
        t0 = time.time()
        doe_export_txt(mc, CASE_DIR, first_step=1, final_step=POINTS, verbose=True)
        doe_export_h5_from_txt(CASE_DIR, verbose=True)
        print(f"TORQUE120_REEXPORT_DONE in {time.time()-t0:.0f}s", flush=True)
    finally:
        try:
            mc.quit()
        except Exception:                                       # noqa: BLE001
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
