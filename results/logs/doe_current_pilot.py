"""R8 pilot: add current-axis levels to the ORIGINAL 40 geometries.

Unlike doe_gen (new geometries from an LHS), this campaign re-solves EXISTING
case geometries at new current levels, so the geometry definition is reused
byte-for-byte from each case's own .mot -- no geometry re-propagation, no mesh
drift risk beyond Motor-CAD's own remesh.

Per source case i in 0..39 (D:/KDH/Sim_4SolverX/DOE4TrainingData/case_XXXX)
and level j in {0.5, 0.25} of the 650.538 A template anchor:

    local case k = 2*i + j  ->  OUT/case_{k:04d}/TestCAD1.mot (copied source .mot)
    set CurrentDefinition=0 (wiring fix, section 26), PeakCurrent=level
    do_magnetic_calculation -> export txt -> txt-to-h5

The local manifest records geometry (from backup/doe_data's manifest, the
values that actually shaped the mesh), electrical = {PeakCurrent=level,
PhaseAdvance=source, CurrentDefinition=0, excitation_wired=1}, and the source
geometry index for the split builder.

Run with the Ansys venv:
  C:/Users/moa/.ansys_python_venvs/PyMotorEnv_310/Scripts/python.exe results/logs/doe_current_pilot.py
Env: PILOT_OUT (default D:/KDH/Sim_4SolverX/DOE_CurrentAxis), PILOT_GEOMS (default 0..39),
     PILOT_LEVELS (default "0.5,0.25"), PILOT_SHARD (manifest shard tag for parallel
     workers), PILOT_MERGE=1 (no solving: rebuild doe_manifest.json from case dirs)

Parallelism and resume: the case index is DETERMINISTIC (k = 2*geom + level_idx),
already-exported cases (postproc/*.h5 present) are skipped, and each worker writes
its own manifest shard — so N workers can each take a PILOT_GEOMS slice with their
own Motor-CAD instance (licence permitting), and a crashed/killed run resumes by
relaunching over the full list. PILOT_MERGE reconstructs the final manifest from
disk, which is safe because every field is derivable from k and the training
manifest.
"""
import json
import os
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

ANCHOR_IPK = 650.538238691624                      # section 26: template 460 A RMS
SRC_ROOT = pathlib.Path(r"D:/KDH/Sim_4SolverX/DOE4TrainingData")
OUT = pathlib.Path(os.environ.get("PILOT_OUT", r"D:/KDH/Sim_4SolverX/DOE_CurrentAxis"))
GEOMS = [int(x) for x in os.environ.get("PILOT_GEOMS", ",".join(map(str, range(40)))).split(",")]
LEVELS = [float(x) for x in os.environ.get("PILOT_LEVELS", "0.5,0.25").split(",")]

TRAIN_MANIFEST = json.loads(
    (REPO / "backup/doe_data/doe_manifest.json").read_text(encoding="utf-8"))
GEOM_OF = {int(c["index"]): c.get("geometry", {}) for c in TRAIN_MANIFEST["cases"]}
ADV_OF = {int(c["index"]): float(c.get("electrical", {}).get("PhaseAdvance", 0.0))
          for c in TRAIN_MANIFEST["cases"]}


def case_entry(k, gi, lv, h5s, solve_s):
    return {
        "index": k,
        "source_geometry_index": gi,
        "geometry": GEOM_OF.get(gi, {}),
        "electrical": {
            "PeakCurrent": ANCHOR_IPK * lv,
            "PhaseAdvance": ADV_OF.get(gi, 0.0),
            "CurrentDefinition": 0,
            "excitation_wired": 1,
        },
        "h5_paths": [str(p) for p in h5s],
        "solve_time_s": solve_s,
    }


def merge_manifest():
    """Rebuild doe_manifest.json from disk — every field is derivable from k."""
    cases = []
    for case_dir in sorted(OUT.glob("case_*")):
        k = int(case_dir.name.split("_")[1])
        gi, j = k // len(LEVELS), k % len(LEVELS)
        h5s = sorted((case_dir / "postproc").glob("*.h5"))
        if not h5s:
            continue
        cases.append(case_entry(k, gi, LEVELS[j], h5s, None))
    (OUT / "doe_manifest.json").write_text(json.dumps(
        {"n_cases": len(cases), "campaign": "current_axis_pilot",
         "anchor_ipk": ANCHOR_IPK, "levels": LEVELS, "cases": cases,
         "failed": []}, indent=1), encoding="utf-8")
    print(f"PILOT_MERGE_DONE n={len(cases)}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if os.environ.get("PILOT_MERGE") == "1":
        merge_manifest()
        return
    shard = os.environ.get("PILOT_SHARD", "")
    manifest_path = OUT / (f"doe_manifest_shard_{shard}.json" if shard else "doe_manifest.json")
    mc = pmc.MotorCAD()
    cases = []
    failed = []
    t_all = time.time()
    try:
        mc.set_variable("MessageDisplayState", 2)
        for gi in GEOMS:
            for j, lv in enumerate(LEVELS):
                k = len(LEVELS) * gi + j          # deterministic across workers
                tag = f"case_{k:04d} (geom {gi}, {lv:.2f}x = {ANCHOR_IPK*lv:.1f} A)"
                case_dir = OUT / f"case_{k:04d}"
                if sorted((case_dir / "postproc").glob("*.h5")):
                    print(f"[pilot] {tag}: SKIP (already exported)", flush=True)
                    continue
                try:
                    src_mot = SRC_ROOT / f"case_{gi:04d}" / "TestCAD1.mot"
                    case_dir.mkdir(parents=True, exist_ok=True)
                    case_mot = case_dir / "TestCAD1.mot"
                    shutil.copy2(src_mot, case_mot)

                    mc.load_from_file(str(case_mot))
                    mc.set_variable("MessageDisplayState", 2)
                    mc.set_variable("CurrentDefinition", 0)      # wiring fix
                    ipk = ANCHOR_IPK * lv
                    mc.set_variable("PeakCurrent", ipk)
                    mc.save_to_file(str(case_mot))
                    try:
                        mc.display_screen("E-Magnetics;FEA")
                    except Exception:
                        pass
                    t0 = time.time()
                    mc.do_magnetic_calculation()
                    solve_s = time.time() - t0
                    mc.save_to_file(str(case_mot))

                    doe_export_txt(mc, case_dir, verbose=False)
                    doe_export_h5_from_txt(case_dir, verbose=False)

                    h5s = sorted((case_dir / "postproc").glob("*.h5"))
                    if not h5s:
                        raise FileNotFoundError("no h5 after export")
                    cases.append(case_entry(k, gi, lv, h5s, round(solve_s, 1)))
                    print(f"[pilot] {tag}: solve {solve_s:.0f}s, {len(h5s)} h5  "
                          f"({time.time()-t_all:.0f}s total)", flush=True)
                except Exception as exc:
                    failed.append({"index": k, "geom": gi, "level": lv, "error": str(exc)})
                    print(f"[pilot] {tag} FAILED: {exc}", flush=True)
                # write manifest incrementally so a crash loses nothing
                manifest_path.write_text(json.dumps(
                    {"n_cases": len(cases), "campaign": "current_axis_pilot",
                     "anchor_ipk": ANCHOR_IPK, "levels": LEVELS,
                     "cases": cases, "failed": failed}, indent=1), encoding="utf-8")
    finally:
        try:
            mc.quit()
        except Exception:
            pass
    print(f"PILOT_GEN_DONE n={len(cases)} failed={len(failed)} in {time.time()-t_all:.0f}s")


if __name__ == "__main__":
    main()
