"""Is the air-gap smoothness residual physics, or is it band discretisation?

Section 31 measured that the best purely angular description of the air-gap field
reaches 5.396% pooled |B| nRMSE on the gate's element set, against a 5% target.
It could not say whether the un-describable part is real -- radial variation
across a 1 mm gap, slot-opening structure -- or an artifact of how Motor-CAD
meshes the band. This decides it.

Single variable: the SAME case file, the SAME excitation, the SAME geometry, only
the air-gap mesh controls change. Then re-run the section-31 measurement.

    AirgapMesh_NumLayers        radial layers  (default 4 -- these are a1..a4)
    AirgapMeshPoints_layers     points around the gap (default 360)
    AirgapMeshPoints_mesh       points on the gap surfaces (default 360)

Reading the result:
  residual FALLS materially  -> discretisation. The target carries mesh noise, so
                                a denser or continuum-style gap treatment in the
                                GENERATOR is a real lever -- and a v4-campaign
                                decision, since it breaks the single generation
                                process sections 13-30 share.
  residual HOLDS             -> physics. No meshing change helps; the last 0.40 pp
                                of the gate is structure the surrogate must learn.

Compare on the ADMISSIBLE and n<=200 rows, not n<=350: those bases have a fixed
parameter count, so they mean the same thing on meshes of different size. The
n<=350 row approaches interpolation and its meaning drifts with element count.

Run with the Ansys venv:
  C:/Users/moa/.ansys_python_venvs/PyMotorEnv_310/Scripts/python.exe results/logs/airgap_mesh_probe.py
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
import time

REPO = pathlib.Path(r"D:/KDH/NvidiaNemo")
sys.path.insert(0, str(REPO / "eMach"))

from tools.motorCAD.pyMCAD.doe_batch import (          # noqa: E402
    doe_export_h5_from_txt,
    doe_export_txt,
)
import ansys.motorcad.core as pmc                       # noqa: E402

SRC_MOT = r"D:/KDH/Sim_4SolverX/DOE4TrainingData/case_0004/TestCAD1.mot"
OUT = pathlib.Path(os.environ.get("PROBE_OUT", r"D:/KDH/Sim_4SolverX/DOE_MeshProbe"))

# (name, layers, points_layers, points_mesh). "baseline" re-solves the shipped
# settings: if it does not reproduce the original measurement the comparison is
# not trustworthy, so it is a control, not a formality.
ARMS = [
    # ROUND 1 (kept for the record; see the note below -- it did not decide anything)
    ("baseline_4x360", 4, 360, 360),
    ("radial_8x360", 8, 360, 360),
    ("circum_4x720", 4, 720, 720),
    ("both_8x720", 8, 720, 720),
    # ROUND 2. Round 1 mislabelled its own control: the shipped case file carries
    # AirgapMeshPoints = 1440, not the 360 default, so "baseline_4x360" was a
    # COARSENING arm and every comparison hung off a control that never reproduced
    # the original 7.808%. Two further findings from round 1: AirgapMesh_NumLayers
    # 4 -> 8 produced a byte-identical mesh (8227 nodes / 16160 elements / 1209 gap
    # elements, same as 4 layers) so that variable is a no-op on this model; and the
    # TOTAL element count moved 13426 -> 16160, i.e. re-saving perturbs the whole
    # mesh, not just the gap. Until the true control reproduces the original number,
    # nothing here can be read as physics-vs-discretisation.
    ("control_4x1440", 4, 1440, 1440),      # the settings the campaign actually used
    ("fine_4x2880", 4, 2880, 2880),         # genuine refinement ABOVE the real baseline
]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    mc = pmc.MotorCAD()
    results = {}
    try:
        mc.set_variable("MessageDisplayState", 2)
        for name, layers, p_layers, p_mesh in ARMS:
            case_dir = OUT / name / "case_0000"
            if sorted((case_dir / "postproc").glob("*.h5")):
                print(f"[{name}] SKIP (already exported)", flush=True)
                results[name] = "skipped"
                continue
            case_dir.mkdir(parents=True, exist_ok=True)
            t0 = time.time()
            mc.load_from_file(SRC_MOT)
            mc.set_variable("MessageDisplayState", 2)
            # ONLY the mesh controls move. Excitation, geometry, materials and the
            # 45-step rotor sweep stay exactly as the campaign solved them.
            mc.set_variable("AirgapMesh_NumLayers", int(layers))
            mc.set_variable("AirgapMeshPoints_layers", int(p_layers))
            mc.set_variable("AirgapMeshPoints_mesh", int(p_mesh))
            mc.save_to_file(str(case_dir / "TestCAD1.mot"))
            try:
                mc.display_screen("E-Magnetics;FEA")
            except Exception:                            # noqa: BLE001
                pass
            mc.do_magnetic_calculation()
            solve_s = time.time() - t0
            try:
                used = int(mc.get_variable("AirgapMesh_NumLayers_Used"))
            except Exception:                            # noqa: BLE001
                used = -1
            mc.save_to_file(str(case_dir / "TestCAD1.mot"))
            doe_export_txt(mc, case_dir, verbose=False)
            doe_export_h5_from_txt(case_dir, verbose=False)
            h5s = sorted((case_dir / "postproc").glob("*.h5"))

            # one-case manifest so airgap_smoothness_floor.py can read this dir
            (OUT / name / "doe_manifest.json").write_text(json.dumps({
                "n_cases": 1, "campaign": f"airgap_mesh_probe_{name}",
                "cases": [{"index": 0, "source_geometry_index": 4,
                           "geometry": {}, "electrical": {},
                           "h5_paths": [str(p) for p in h5s]}],
            }, indent=1), encoding="utf-8")
            results[name] = {"solve_s": round(solve_s), "layers_requested": layers,
                             "layers_used": used, "points": p_layers, "n_h5": len(h5s)}
            print(f"[{name}] solve {solve_s:.0f}s, layers used {used}, {len(h5s)} h5",
                  flush=True)
    finally:
        try:
            mc.quit()
        except Exception:                                # noqa: BLE001
            pass

    (OUT / "probe_summary.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    print("AIRGAP_MESH_PROBE_DONE", json.dumps(results), flush=True)
    print("\nNow measure each arm:")
    for name, *_ in ARMS:
        print(f"  python results/logs/airgap_smoothness_floor.py "
              f"--data-dir {OUT / name} --cases 0 --out results/airgap_floor_{name}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
