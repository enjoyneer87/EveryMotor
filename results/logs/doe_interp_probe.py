"""Current-interpolation probe: the 6 legacy TEST geometries at 487.904 A (0.75x).

EVAL-ONLY by design (section 30): these cases never enter any training set. They ask
the one question section 29 could not -- does the model interpolate to a current LEVEL
it never saw (trained levels: 650.5 / 325.3 / 162.6 A)? Pre-registered reading: pooled
torque nRMSE landing between the 325.3 A group (2.404%) and the 650.5 A legacy gate
(3.270%) counts as interpolation holding; no gate.

Pilot pattern (copy source .mot, set current, solve, export). Run AFTER the Ext3
workers finish -- one more Motor-CAD instance during generation would fight for CPU.

  C:/Users/moa/.ansys_python_venvs/PyMotorEnv_310/Scripts/python.exe results/logs/doe_interp_probe.py
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

ANCHOR_IPK = 650.538238691624
LEVEL = 0.75
GEOMS = [4, 7, 18, 32, 37, 39]                 # the legacy test geometries
SRC_ROOT = pathlib.Path(r"D:/KDH/Sim_4SolverX/DOE4TrainingData")
OUT = pathlib.Path(r"D:/KDH/Sim_4SolverX/DOE_InterpProbe")

TRAIN_MANIFEST = json.loads(
    (REPO / "backup/doe_data/doe_manifest.json").read_text(encoding="utf-8"))
GEOM_OF = {int(c["index"]): c.get("geometry", {}) for c in TRAIN_MANIFEST["cases"]}
ADV_OF = {int(c["index"]): float(c.get("electrical", {}).get("PhaseAdvance", 0.0))
          for c in TRAIN_MANIFEST["cases"]}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    ipk = ANCHOR_IPK * LEVEL
    mc = pmc.MotorCAD()
    cases, failed = [], []
    t_all = time.time()
    try:
        mc.set_variable("MessageDisplayState", 2)
        for i, gi in enumerate(GEOMS):
            tag = f"case_{i:04d} (geom {gi}, {ipk:.1f} A)"
            case_dir = OUT / f"case_{i:04d}"
            if sorted((case_dir / "postproc").glob("*.h5")):
                print(f"[probe] {tag}: SKIP", flush=True)
                continue
            try:
                src_mot = SRC_ROOT / f"case_{gi:04d}" / "TestCAD1.mot"
                case_dir.mkdir(parents=True, exist_ok=True)
                case_mot = case_dir / "TestCAD1.mot"
                shutil.copy2(src_mot, case_mot)
                mc.load_from_file(str(case_mot))
                mc.set_variable("MessageDisplayState", 2)
                mc.set_variable("CurrentDefinition", 0)      # section 26 fix
                mc.set_variable("PeakCurrent", ipk)
                mc.save_to_file(str(case_mot))
                t0 = time.time()
                mc.do_magnetic_calculation()
                solve_s = time.time() - t0
                mc.save_to_file(str(case_mot))
                doe_export_txt(mc, case_dir, verbose=False)
                doe_export_h5_from_txt(case_dir, verbose=False)
                h5s = sorted((case_dir / "postproc").glob("*.h5"))
                cases.append({
                    "index": i, "source_geometry_index": gi,
                    "geometry": GEOM_OF.get(gi, {}),
                    "electrical": {"PeakCurrent": ipk, "PhaseAdvance": ADV_OF.get(gi, 0.0),
                                   "CurrentDefinition": 0, "excitation_wired": 1},
                    "h5_paths": [str(p) for p in h5s], "solve_time_s": solve_s,
                })
                print(f"[probe] {tag}: solve {solve_s:.0f}s ({time.time()-t_all:.0f}s total)",
                      flush=True)
            except Exception as exc:                         # noqa: BLE001
                failed.append({"case": i, "error": repr(exc)})
                print(f"[probe] {tag}: FAILED {exc!r}", flush=True)
    finally:
        try:
            mc.quit()
        except Exception:                                    # noqa: BLE001
            pass
    (OUT / "doe_manifest.json").write_text(json.dumps(
        {"n_cases": len(cases), "campaign": "interp_probe_487A_eval_only",
         "anchor_ipk": ANCHOR_IPK, "level": LEVEL, "cases": cases, "failed": failed},
        indent=1), encoding="utf-8")
    print(f"INTERP_PROBE_DONE n={len(cases)} failed={len(failed)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
