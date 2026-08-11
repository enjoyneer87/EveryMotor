"""Export a campaign case from Motor-CAD to a Maxwell 2D model.

Step 1 of the Continuum Air (CA) on/off comparison that section 31d leaves as the
only experiment able to settle what the residual 6.95% on layer a1 is.

Why the comparison has to live in Maxwell: CA is a Maxwell beta feature. Section
31c confirmed it exists for BOTH 2D and 3D in the installed 2026 R1
(Maxwell.pdf p.4264 "Continuum Air 2D", p.4189 "Continuum Air 3D"), and that what
it fixes is field mapping across the nonconformal moving/stationary interface --
the one mechanism our mesh-refinement probe provably cannot test, because an
interpolation artifact does not wash out under refinement.

The valid comparison is CA on vs CA off on the SAME Maxwell model. Maxwell vs
Motor-CAD would confound the solver with the feature.

Pipeline and who does what:
  1. (this script)  Motor-CAD -> Maxwell 2D export script
  2. (USER, once)   AEDT: Tools > Options > General Options > Beta Options >
                    Continuum Air 2D > OK > restart. It is a GUI toggle and it is
                    not written to the personal .cfg until first used, so it
                    cannot be set reliably from here.
  3. (scriptable)   run the exported script in AEDT to build the model
  4. (scriptable)   solve transient twice, CA off then on
  5. (scriptable)   export element B, feed results/logs/airgap_layer_breakdown.py

Run with the Ansys venv:
  C:/Users/moa/.ansys_python_venvs/PyMotorEnv_310/Scripts/python.exe results/logs/mcad_to_maxwell_export.py
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
import time

REPO = pathlib.Path(r"D:/KDH/NvidiaNemo")
sys.path.insert(0, str(REPO / "eMach"))

import ansys.motorcad.core as pmc                       # noqa: E402

SRC_MOT = os.environ.get("SRC_MOT", r"D:/KDH/Sim_4SolverX/DOE4TrainingData/case_0004/TestCAD1.mot")
OUT = pathlib.Path(os.environ.get("MXW_OUT", r"D:/KDH/Sim_4SolverX/MaxwellExport_case4"))

# Motor-CAD Ansys-export settings (names from the ActiveX parameter catalogue).
SETTINGS = {
    "AnsysModelType": 0,        # 0 = 2D
    "Ansys_ScriptFormat": 0,    # 0 = Python script (readable, and drivable by pyaedt)
    "AnsysAirgapMesh": 1,       # keep the cylindrical air-gap mesh on the export
}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    mc = pmc.MotorCAD()
    info = {"src_mot": SRC_MOT, "out": str(OUT), "settings": {}}
    try:
        mc.set_variable("MessageDisplayState", 2)
        mc.load_from_file(SRC_MOT)
        mc.set_variable("MessageDisplayState", 2)
        for k, v in SETTINGS.items():
            try:
                mc.set_variable(k, v)
                info["settings"][k] = v
            except Exception as exc:                    # noqa: BLE001
                info["settings"][k] = f"FAILED {exc!r}"
                print(f"  set {k}={v} failed: {exc!r}", flush=True)
        # record what the gap mesh actually is, so the Maxwell side can be matched
        for probe in ("AirgapMeshPoints_layers", "AirgapMeshPoints_mesh",
                      "AirgapMesh_NumLayers", "AirgapMesh_NumLayers_Used"):
            try:
                info[probe] = mc.get_variable(probe)
            except Exception:                           # noqa: BLE001
                info[probe] = None

        t0 = time.time()
        target = str(OUT / "case4_maxwell2d")
        mc.export_to_ansys_electronics_desktop(target)
        info["export_s"] = round(time.time() - t0, 1)
        info["files"] = sorted(p.name for p in OUT.iterdir())
        print(f"export ok in {info['export_s']}s -> {OUT}", flush=True)
        for f in info["files"]:
            print("   ", f, flush=True)
    finally:
        try:
            mc.quit()
        except Exception:                               # noqa: BLE001
            pass
    (OUT / "export_info.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    print("MCAD_TO_MAXWELL_EXPORT_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
