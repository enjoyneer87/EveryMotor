"""480-case expansion, batch 3: +120 new geometries at the legacy excitation
+ 20 of them at two lower current levels (section 29 design).

Design (recorded in methodology section 30):
  local k = 0..119   new geometry g=k (3D LHS: Ratio_Bore, Ratio_SlotDepth, PhaseAdvance;
                     seed 46, same bounds as the original space -- densify, no widening),
                     PeakCurrent = 650.538238691624 A (the legacy-gate excitation)
  local k = 120..159 factorial extension: for j = 0..19, geometry row 6*j at
                     k=120+2j -> 325.269 A, k=121+2j -> 162.635 A.
                     Breaks the "current variation only on the original 40
                     geometries" confound without spending the geometry budget.
  TorquePointsPerCycle stays 45: section 28 rejected mixed step counts, and this
  batch must move ONLY the data axis (single-axis discipline, sections 13-21).
  CurrentDefinition=0 on every case (section 26 wiring fix); the A-turns gate
  (doe_verify_excitation) runs after ingest.

Parallel + resumable, pilot pattern: case index is deterministic, cases whose
postproc/*.h5 exist are skipped, worker w of N takes k % N == w, each worker has
its own Motor-CAD instance and manifest shard; GEN_MERGE=1 rebuilds the final
manifest from disk.

Run one worker per console with the Ansys venv:
  $env:GEN_WORKER=0; $env:GEN_WORKERS=6
  C:/Users/moa/.ansys_python_venvs/PyMotorEnv_310/Scripts/python.exe results/logs/doe_gen_480.py
"""
import json
import os
import pathlib
import sys
import time

REPO = pathlib.Path(r"D:/KDH/NvidiaNemo")
sys.path.insert(0, str(REPO / "eMach"))

from tools.pyutils.sweep import DOEAxis, build_doe_lhs          # noqa: E402
from tools.motorCAD.pyMCAD.doe_batch import (                   # noqa: E402
    doe_export_txt,
    doe_export_h5_from_txt,
)
from tools.motorCAD.pyMCAD.melec_req_check import set_mcad_variables   # noqa: E402
from tools.motorCAD.pyMCAD.winding_auto import (                # noqa: E402
    auto_resize_copper_after_geometry_update,
)
import ansys.motorcad.core as pmc                               # noqa: E402

# PIPELINE REV 2 (2026-08-09): rev 1 applied geometry ratios with raw set_variable and
# skipped auto_resize_copper_after_geometry_update -- the conductor-dimension recompute
# the proven doe_batch pipeline runs after every geometry change. For 19 of 120 LHS
# geometries the stale conductor layout became infeasible and Motor-CAD silently dropped
# every Turn_* region: the case solved with j = 0 (magnet-only field) while exporting
# cleanly. The ampere-turns gate caught it (23 NaN cases = those 19 geometries + their
# factorial variants). The whole batch was regenerated under rev 2 so all 480 training
# cases share ONE generation process (the 137 rev-1 survivors carried template conductor
# dimensions, a different process from the auto-resized 240 base).

BASE_MOT = r"D:/KDH/Sim_4SolverX/TestCAD1.mot"
OUT = pathlib.Path(os.environ.get("GEN_OUT", r"D:/KDH/Sim_4SolverX/DOE_Ext3"))
SEED = 46
N_GEOM = 120
N_FACT = 20                       # every 6th geometry gets the two lower levels
ANCHOR_IPK = 650.538238691624     # section 26: template 460 A RMS -> peak
LEVELS = (0.5 * ANCHOR_IPK, 0.25 * ANCHOR_IPK)
WORKER = int(os.environ.get("GEN_WORKER", "0"))
WORKERS = int(os.environ.get("GEN_WORKERS", "1"))


def build_plan(mc=None):
    """Deterministic 160-row plan; identical in every worker (fixed seed)."""
    # Base ratios come from the template so the bounds match doe_gen.py exactly.
    if mc is None:
        probe = pmc.MotorCAD()
        try:
            probe.load_from_file(BASE_MOT)
            rb = float(probe.get_variable("Ratio_Bore"))
            rsd = float(probe.get_variable("Ratio_SlotDepth_ParallelSlot"))
        finally:
            probe.quit()
    else:
        mc.load_from_file(BASE_MOT)
        rb = float(mc.get_variable("Ratio_Bore"))
        rsd = float(mc.get_variable("Ratio_SlotDepth_ParallelSlot"))

    axes = [
        DOEAxis("Ratio_Bore", round(rb * 0.90, 4), round(rb * 1.10, 4), steps=0),
        DOEAxis("Ratio_SlotDepth_ParallelSlot", round(rsd * 0.85, 4), round(rsd * 1.15, 4), steps=0),
        DOEAxis("PhaseAdvance", 0.0, 90.0, steps=0),
    ]
    rows = build_doe_lhs(axes=axes, n_samples=N_GEOM, seed=SEED, criterion="maximin")

    plan = []
    for g, pt in enumerate(rows):
        vals = {**pt.geometry, **pt.electrical}          # LHS spreads across both dicts
        geom = {k: vals[k] for k in ("Ratio_Bore", "Ratio_SlotDepth_ParallelSlot")}
        adv = vals["PhaseAdvance"]
        plan.append({"k": g, "geom_row": g, "geometry": geom,
                     "PeakCurrent": ANCHOR_IPK, "PhaseAdvance": adv})
    for j in range(N_FACT):
        src = plan[6 * j]
        for lidx, ipk in enumerate(LEVELS):
            plan.append({"k": 120 + 2 * j + lidx, "geom_row": src["geom_row"],
                         "geometry": src["geometry"], "PeakCurrent": ipk,
                         "PhaseAdvance": src["PhaseAdvance"]})
    assert len(plan) == 160 and [p["k"] for p in plan] == list(range(160))
    return plan


def case_entry(p, h5s, solve_s):
    return {
        "index": p["k"],
        "source_geometry_index": p["geom_row"],       # row in THIS batch's LHS table
        "geometry": p["geometry"],
        "electrical": {"PeakCurrent": p["PeakCurrent"], "PhaseAdvance": p["PhaseAdvance"],
                       "CurrentDefinition": 0, "excitation_wired": 1},
        "h5_paths": [str(x) for x in h5s],
        "solve_time_s": solve_s,
    }


def merge_manifest():
    plan = build_plan()
    cases, missing = [], []
    for p in plan:
        h5s = sorted((OUT / f"case_{p['k']:04d}" / "postproc").glob("*.h5"))
        if h5s:
            cases.append(case_entry(p, h5s, None))
        else:
            missing.append(p["k"])
    (OUT / "doe_manifest.json").write_text(json.dumps(
        {"n_cases": len(cases), "campaign": "ext3_480_expansion",
         "seed": SEED, "anchor_ipk": ANCHOR_IPK,
         "design": "120 new geoms @650.538A + 20 of them x {325.269, 162.635}A",
         "cases": cases, "missing": missing}, indent=1), encoding="utf-8")
    print(f"GEN480_MERGE_DONE n={len(cases)} missing={missing}", flush=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if os.environ.get("GEN_MERGE") == "1":
        merge_manifest()
        return

    mc = pmc.MotorCAD()
    plan = build_plan(mc)
    mine = [p for p in plan if p["k"] % WORKERS == WORKER]
    print(f"[gen480 w{WORKER}/{WORKERS}] {len(mine)} cases", flush=True)
    shard, failed = [], []
    t_all = time.time()
    try:
        mc.set_variable("MessageDisplayState", 2)
        for p in mine:
            k = p["k"]
            tag = (f"case_{k:04d} (geom {p['geom_row']}, {p['PeakCurrent']:.1f} A, "
                   f"adv {p['PhaseAdvance']:.2f})")
            case_dir = OUT / f"case_{k:04d}"
            if sorted((case_dir / "postproc").glob("*.h5")):
                print(f"[gen480 w{WORKER}] {tag}: SKIP (already exported)", flush=True)
                continue
            try:
                case_dir.mkdir(parents=True, exist_ok=True)
                case_mot = case_dir / "TestCAD1.mot"
                mc.load_from_file(BASE_MOT)
                mc.set_variable("MessageDisplayState", 2)
                all_vars = {**{k: float(v) for k, v in p["geometry"].items()},
                            "CurrentDefinition": 0,              # section 26 fix
                            "PeakCurrent": float(p["PeakCurrent"]),
                            "PhaseAdvance": float(p["PhaseAdvance"])}
                set_mcad_variables(mc, all_vars, verbose=False)
                # The step rev 1 fatally skipped: recompute conductor dimensions for
                # the resized slot, exactly as the proven doe_batch pipeline does.
                auto_resize_copper_after_geometry_update(mc, all_vars=all_vars, verbose=False)
                mc.save_to_file(str(case_mot))
                try:
                    mc.display_screen("E-Magnetics;FEA")
                except Exception:                                # noqa: BLE001
                    pass
                t0 = time.time()
                mc.do_magnetic_calculation()
                solve_s = time.time() - t0
                mc.save_to_file(str(case_mot))
                doe_export_txt(mc, case_dir, verbose=False)
                doe_export_h5_from_txt(case_dir, verbose=False)
                h5s = sorted((case_dir / "postproc").glob("*.h5"))
                shard.append(case_entry(p, h5s, solve_s))
                print(f"[gen480 w{WORKER}] {tag}: solve {solve_s:.0f}s, {len(h5s)} h5  "
                      f"({time.time()-t_all:.0f}s total)", flush=True)
            except Exception as exc:                             # noqa: BLE001
                failed.append({"k": k, "error": repr(exc)})
                print(f"[gen480 w{WORKER}] {tag}: FAILED {exc!r}", flush=True)
    finally:
        try:
            mc.quit()
        except Exception:                                        # noqa: BLE001
            pass
    (OUT / f"doe_manifest_shard_w{WORKER}.json").write_text(
        json.dumps({"worker": WORKER, "cases": shard, "failed": failed}, indent=1),
        encoding="utf-8")
    print(f"GEN480_WORKER{WORKER}_DONE ok={len(shard)} failed={len(failed)} "
          f"in {time.time()-t_all:.0f}s", flush=True)


if __name__ == "__main__":
    main()
