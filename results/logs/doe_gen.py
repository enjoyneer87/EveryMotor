"""R4(a) DOE expansion -- generate N new motor geometries via Motor-CAD (PyMotorEnv_310).

Isolated output root (never touches the existing 40 in DOE_TrainingData). Case indices
0..N-1 inside OUT_ROOT; ingestion renumbers to case_0040+. Env knobs:
  DOE_N (default 2), DOE_SEED (43), DOE_OUT (D:/KDH/Sim_4SolverX/DOE_Ext), DOE_WORKERS (1)

Run with the Ansys venv:
  C:\\Users\\moa\\.ansys_python_venvs\\PyMotorEnv_310\\Scripts\\python.exe results/logs/doe_gen.py
"""
import json
import os
import sys
import pathlib

REPO = pathlib.Path(r"D:/KDH/NvidiaNemo")
sys.path.insert(0, str(REPO / "eMach"))

from tools.pyutils.sweep import DOEAxis, build_doe_lhs
from tools.motorCAD.pyMCAD.doe_batch import doe_batch_run
import ansys.motorcad.core as pmc

BASE_MOT = r"D:/KDH/Sim_4SolverX/TestCAD1.mot"
N = int(os.environ.get("DOE_N", "2"))
SEED = int(os.environ.get("DOE_SEED", "43"))
OUT = os.environ.get("DOE_OUT", r"D:/KDH/Sim_4SolverX/DOE_Ext")
WORKERS = int(os.environ.get("DOE_WORKERS", "1"))

print(f"[doe_gen] N={N} seed={SEED} out={OUT} workers={WORKERS} base={BASE_MOT}", flush=True)

mc = pmc.MotorCAD()
try:
    mc.load_from_file(BASE_MOT)
    rb = float(mc.get_variable("Ratio_Bore"))
    rsd = float(mc.get_variable("Ratio_SlotDepth_ParallelSlot"))
    print(f"[doe_gen] base Ratio_Bore={rb:.4f} Ratio_SlotDepth={rsd:.4f}", flush=True)

    # SAME 4D space as the existing 40-case LHS (densify, no widening).
    axes = [
        DOEAxis("Ratio_Bore", round(rb * 0.90, 4), round(rb * 1.10, 4), steps=0),
        DOEAxis("Ratio_SlotDepth_ParallelSlot", round(rsd * 0.85, 4), round(rsd * 1.15, 4), steps=0),
        DOEAxis("PeakCurrent", 10.0, 650.53, steps=0),
        DOEAxis("PhaseAdvance", 0.0, 90.0, steps=0),
    ]
    grid = build_doe_lhs(axes=axes, n_samples=N, seed=SEED, criterion="maximin")

    # EXCITATION WIRING FIX (methodology review section 26): the template is in
    # RMS current-definition mode (CurrentDefinition=1), where PeakCurrent is a
    # DERIVED display quantity -- every pre-2026-08-05 DOE case solved at the
    # template's RMSCurrent=460 regardless of the PeakCurrent axis. Forcing
    # peak mode per case makes the axis live. Verified causally: in mode 0,
    # PeakCurrent=650.538 reproduces the RMS-mode baseline torque and
    # PeakCurrent=325.269 reproduces the RMSCurrent=230 solve.
    for pt in grid:
        pt.electrical["CurrentDefinition"] = 0
    if grid:
        print(f"[doe_gen] built {len(grid)} LHS points; first geom={grid[0].geometry} elec={grid[0].electrical}", flush=True)

    manifest = doe_batch_run(
        mc, grid,
        base_mot=BASE_MOT,
        doe_out_root=OUT,
        phases=("solve", "export"),
        parallel_workers=WORKERS,
        verbose=True,
    )
    n_ok = manifest.get("n_cases", len(grid))
    print(f"[doe_gen] DONE n_cases={n_ok} failed={manifest.get('failed')}", flush=True)
    # write a small summary next to the output for the ingest step
    (REPO / "results/logs/doe_gen_summary.json").write_text(
        json.dumps({"n": N, "seed": SEED, "out": OUT, "n_cases": n_ok,
                    "failed": manifest.get("failed", [])}, indent=2), encoding="utf-8")
finally:
    try:
        mc.quit()
    except Exception:
        pass
