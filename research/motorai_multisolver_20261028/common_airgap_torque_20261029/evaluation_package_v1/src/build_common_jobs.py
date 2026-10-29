"""Build a private, hash-pinned 12-job evaluation package; never launches jobs.

Reads saved metadata only. --release requires explicit admitted geometry and a
PASS CPU checkpoint/source preflight. Without it the package is DRAFT/HOLD.
The output directory must not exist. Remote deployment remains the owner's job.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil

RUN_IDS = tuple(f"{arm}_s{seed}" for arm in ("A0", "A2", "C", "G") for seed in (17, 29, 43))
DEFAULT_STAGE = "C:/work/EveryMotor_CommonA1_20261029"
DEFAULT_PYTHON = "C:/work/JEET_MGN_20261027/.venv/Scripts/python.exe"
DEFAULT_BUNDLE = "C:/work/EveryMotor_FrozenFieldDiag_20261028/bundle"
CG_STAGE = "C:/work/EveryMotor_ContextCG_20261028"
CG_PLAN_SHA = "5a09c1923f5ab4629469a95fede3dc149b56a25274240aabefdc0a60416e7fc2"
COMPONENT_SHA = "79b7f33e4c971723dd88d1b4c74c7864376621fec67b9d605e2a500059eec33f"
EXPECTED_MODULES = {
    "strict_airgap": "src/strict_airgap.py", "dual_solver_mgn": "src/dual_solver_mgn.py",
    "context_field_model": "src/context_field_model.py", "context_checkpoint": "src/context_checkpoint.py",
    "eval": "src/eval/__init__.py", "eval.case_split": "src/eval/case_split.py",
    "eval.feature_guard": "src/eval/feature_guard.py", "eval.mesh_regions": "src/eval/mesh_regions.py",
    "eval.torque": "src/eval/torque.py",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(4 * 1024**2): digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def remote_path(stage, relative):
    relative = relative.replace("\\", "/")
    parts = PurePosixPath(relative)
    require(not parts.is_absolute() and ".." not in parts.parts and ":" not in relative and bool(relative), "unsafe remote relative path")
    return stage.rstrip("/\\") + "/" + str(parts)


def checkpoint_paths():
    result = {
        "A0_s17": "C:/work/JEET_FIELD_SCALE360_TRAIN_20261027/result/last.pt",
        "A0_s29": "C:/work/EveryMotor_L12_Phase45_20261028_R2P/runs/A0_s29/train/last.pt",
        "A0_s43": "C:/work/EveryMotor_L12_A2_20261028/runs/A0_s43/train/last.pt",
    }
    for run in RUN_IDS:
        if run.startswith("A2"):
            result[run] = f"C:/work/EveryMotor_L12_A2_20261028/runs/{run}/train/last.pt"
        elif run[0] in "CG":
            result[run] = f"{CG_STAGE}/runs/{run}/train/last.pt"
    return result


def pinned_local(spec):
    path = Path(spec["local_path"]).resolve(strict=True)
    require(path.is_file() and sha(path) == spec["sha256"], f"local SHA mismatch: {path.name}")
    return path


def validate_preflight(preflight, inventory_runs, mappings):
    require(preflight["status"] == "PASS" and preflight["scope"] == "CPU_METADATA_AND_SAVED_SOURCE_ONLY", "CPU preflight is not PASS")
    rows = preflight["records"]
    require(len(rows) == 12 and {r["id"] for r in rows} == set(RUN_IDS), "CPU preflight must cover twelve exact runs")
    records = {r["id"]: r for r in rows}
    for run in RUN_IDS:
        row = records[run]; arm = inventory_runs[run]["arm"]
        require(row["checkpoint_sha256"] == inventory_runs[run]["eval_recorded_hashes"]["checkpoint_sha256"], f"preflight checkpoint SHA mismatch: {run}")
        require(row["checkpoint_path"].replace("\\", "/").lower() == mappings[run].replace("\\", "/").lower(), f"preflight checkpoint path mismatch: {run}")
        require(row["phase_encoding_resolved"] == ("electrical" if arm == "A2" else "mechanical"), f"preflight phase mismatch: {run}")
        missing = set(row["metadata_missing_fields"])
        allowed = {"A0_s17": {"arm", "phase_encoding"}, "A0_s29": {"phase_encoding"}}.get(run, set())
        require(missing <= allowed, f"unapproved metadata absence: {run}")
        if missing:
            evidence = row["legacy_evidence"]
            require(evidence["status"] in ("PASS", "PASS_CODE_AND_PLAN_HASH")
                    and evidence["checkpoint_sha256"] == row["checkpoint_sha256"]
                    and set(evidence["allowed_absent_fields"]) == missing, f"legacy evidence not hash-bound PASS: {run}")
    return records


def build_package(inventory_path, research_root, runtime_spec_path, package_dir, deadline_utc, release=False):
    inventory_path = Path(inventory_path).resolve(strict=True)
    research_root = Path(research_root).resolve(strict=True)
    runtime_spec_path = Path(runtime_spec_path).resolve(strict=True)
    inventory, runtime = read_json(inventory_path), read_json(runtime_spec_path)
    require(inventory["status"] == "PASS_SAVED_METADATA_ONLY", "saved inventory is not PASS")
    require(len(inventory["runs"]) == 12 and {r["id"] for r in inventory["runs"]} == set(RUN_IDS), "exact twelve inventory runs required")
    runs = {r["id"]: r for r in inventory["runs"]}
    require(len({r["eval_recorded_hashes"]["checkpoint_sha256"] for r in runs.values()}) == 12, "checkpoint reuse across runs")
    deadline = dt.datetime.fromisoformat(deadline_utc)
    require(deadline.tzinfo is not None, "deadline must include UTC offset")
    require(deadline.utcoffset() == dt.timedelta(0), "freeze deadline in UTC")
    stage = runtime.get("stage_root", DEFAULT_STAGE).replace("\\", "/").rstrip("/")
    require(stage == DEFAULT_STAGE, "unapproved remote stage root")
    python = runtime.get("python_executable", DEFAULT_PYTHON)
    bundle = runtime.get("bundle_dir", DEFAULT_BUNDLE)
    mappings = checkpoint_paths()
    geometry_file = pinned_local(runtime["geometry_manifest"])
    geometry = read_json(geometry_file)
    require(geometry["status"] == "ADMITTED" and len(geometry["records"]) == 32
            and len({r["case_index"] for r in geometry["records"]}) == 32
            and all(r["status"] == "ADMITTED" and r["steps_checked"] == 45 for r in geometry["records"]), "dev32 geometry is not admitted")
    common = inventory["verified_common"]
    require(geometry["selection_sha256"] == common["selection_sha256"]
            and geometry["bundle_manifest_sha256"] == common["dev_bundle_manifest_sha256"], "geometry/common bundle mismatch")
    preflight_file = pinned_local(runtime["preflight"])
    preflight_rows = validate_preflight(read_json(preflight_file), runs, mappings)
    require(runtime["cg_plan"]["sha256"] == CG_PLAN_SHA, "CG plan is not original frozen plan")
    cg_plan_file = pinned_local(runtime["cg_plan"])
    component = runtime.get("components_manifest", {"path": f"{CG_STAGE}/components/COMPONENT_MANIFEST.json", "sha256": COMPONENT_SHA})
    require(component["sha256"] == COMPONENT_SHA, "component manifest differs from frozen training")
    source_specs = runtime["source_files"]
    require(len({s["relative"] for s in source_specs}) == len(source_specs), "duplicate source destination")
    module_mapping = {s["module"]: s["relative"] for s in source_specs if "module" in s}
    require(all(module_mapping.get(name) == relative for name, relative in EXPECTED_MODULES.items()), "missing or ambiguous source module pin")
    require("src/evaluate_common_airgap.py" in {s["relative"] for s in source_specs}, "evaluator source missing")
    for source in source_specs:
        require(source["relative"].startswith("src/"), "source outside src")
        remote_path(stage, source["relative"]); pinned_local(source)
    # Validate every saved reference before constructing an output package.
    for run in RUN_IDS:
        for key in ("eval_result", "train_result"):
            record = runs[run][key]
            source = (research_root / record["path"]).resolve(strict=True)
            require(source.is_relative_to(research_root) and sha(source) == record["sha256"], f"saved {run}/{key} changed")
    selection = research_root / "phase2_multigeometry_turns_20261027/SCALE360_SELECTION_FROZEN.json"
    require(sha(selection) == common["selection_sha256"], "frozen selection changed")
    package = Path(package_dir).resolve()
    package.mkdir(parents=True, exist_ok=False)
    def copy(local, relative):
        target = package / PurePosixPath(relative)
        require(not target.exists(), f"package destination already exists: {relative}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(local, target)
        require(sha(target) == sha(local), "copy hash differs")
        return {"path": remote_path(stage, relative), "sha256": sha(target), "bytes": target.stat().st_size}
    def write(value, relative):
        target = package / PurePosixPath(relative)
        require(not target.exists(), f"package destination already exists: {relative}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        return {"path": remote_path(stage, relative), "sha256": sha(target), "bytes": target.stat().st_size}
    source_files = []
    for spec in source_specs:
        row = copy(pinned_local(spec), spec["relative"])
        if "module" in spec: row["module"] = spec["module"]
        source_files.append(row)
    for filename in ("build_common_jobs.py", "guard_common_eval.py"):
        require(f"src/{filename}" not in {s["relative"] for s in source_specs}, "builder/guard auto-pinned; omit from source_files")
        source_files.append(copy(Path(__file__).parent / filename, "src/" + filename))
    pinned_selection = copy(selection, "metadata/SCALE360_SELECTION_FROZEN.json")
    pinned_inventory = copy(inventory_path, "metadata/CHECKPOINT_INVENTORY.json")
    pinned_preflight = copy(preflight_file, "metadata/CPU_PREFLIGHT.json")
    pinned_geometry_receipt = copy(geometry_file, "metadata/ADMITTED_GEOMETRY_COPY.json")
    pinned_cg_plan = copy(cg_plan_file, "metadata/CONTEXT_CG_PLAN.json")
    jobs = []
    for run in RUN_IDS:
        record, observed = runs[run], preflight_rows[run]
        metadata = {}
        for key, name in (("eval_result", "original_eval.json"), ("train_result", "original_train.json")):
            metadata[key] = copy(research_root / record[key]["path"], f"metadata/{run}/{name}")
        job = {
            "schema_version": 1, "id": run, "arm": record["arm"], "seed": record["seed"],
            "release_status": "RELEASED_FROZEN_EVALUATION" if release else "DRAFT_HOLD",
            "deadline_UTC": deadline_utc, "max_seconds": 1800, "workers": 1, "retry_limit": 0,
            "training_updates": 0, "new_FEA": 0, "new_prior_solves": 0,
            "stop_files": [remote_path(stage, "STOP"), remote_path(stage, "runs/STOP")],
            "checkpoint": {"path": mappings[run], "sha256": record["eval_recorded_hashes"]["checkpoint_sha256"]},
            "phase_encoding": observed["phase_encoding_resolved"],
            "source_files": source_files, "selection": pinned_selection, "bundle_dir": bundle,
            "bundle_manifest_sha256": common["dev_bundle_manifest_sha256"],
            "geometry_manifest": {"path": runtime["geometry_manifest"]["path"], "sha256": runtime["geometry_manifest"]["sha256"]},
            "original_eval": metadata["eval_result"], "original_train": metadata["train_result"],
            "metadata_hashes": record["train_recorded_hashes"],
            "legacy_absent_fields": observed["metadata_missing_fields"],
            "preflight_receipt": pinned_preflight,
        }
        if observed["metadata_missing_fields"]:
            job["legacy_evidence"] = write({
                "id": run, "checkpoint_sha256": record["eval_recorded_hashes"]["checkpoint_sha256"],
                "allowed_absent_fields": observed["metadata_missing_fields"],
                "preflight_sha256": pinned_preflight["sha256"], "evidence": observed["legacy_evidence"],
                "phase_encoding_resolved": observed["phase_encoding_resolved"],
            }, f"metadata/{run}/LEGACY_METADATA_EVIDENCE.json")
        if record["arm"] in ("C", "G"):
            job["cg_plan"] = pinned_cg_plan; job["components_manifest"] = component
        job_pin = write(job, f"jobs/{run}.json")
        jobs.append({"id": run, "job": job_pin, "out": remote_path(stage, f"runs/{run}")})
    plan = {
        "schema_version": 1, "id": "COMMON-A1-TORQUE-20261029", "status": "RELEASED_FROZEN_EVALUATION" if release else "DRAFT_HOLD",
        "stage_root": stage, "python_executable": python, "pythonpath": remote_path(stage, "src"),
        "evaluator": next(s for s in source_files if s["path"].endswith("/evaluate_common_airgap.py")),
        "guard": next(s for s in source_files if s["path"].endswith("/guard_common_eval.py")),
        "source_files": source_files, "deadline_UTC": deadline_utc, "global_max_seconds": 7200,
        "per_model_max_seconds": 1800, "workers": 1, "retry_limit": 0,
        "training_updates": 0, "new_FEA": 0, "new_prior_solves": 0,
        "runs_root": remote_path(stage, "runs"), "stop_files": [remote_path(stage, "STOP"), remote_path(stage, "runs/STOP")],
        "preflight_receipt": pinned_preflight, "inventory_receipt": pinned_inventory,
        "geometry_manifest": {"path": runtime["geometry_manifest"]["path"], "sha256": runtime["geometry_manifest"]["sha256"]},
        "geometry_receipt_copy": pinned_geometry_receipt,
        "disk_free_min_bytes": int(runtime.get("disk_free_min_bytes", 5 * 1024**3)),
        "jobs": jobs, "actual_jobs_launched_by_builder": 0,
    }
    plan_pin = write(plan, "BATCH_PLAN.json")
    manifest = [{"relative": p.relative_to(package).as_posix(), "sha256": sha(p), "bytes": p.stat().st_size}
                for p in sorted(package.rglob("*")) if p.is_file()]
    write({"files": manifest, "batch_plan_sha256": plan_pin["sha256"], "remote_execution": False}, "PACKAGE_MANIFEST.json")
    return {"package_dir": str(package), "batch_plan": plan_pin, "jobs": 12, "release_status": plan["status"], "launched": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--research-root", type=Path, required=True)
    parser.add_argument("--runtime-spec", type=Path, required=True)
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--deadline-utc", required=True)
    parser.add_argument("--release", action="store_true")
    a = parser.parse_args()
    print(json.dumps(build_package(a.inventory, a.research_root, a.runtime_spec, a.package_dir, a.deadline_utc, a.release), indent=2))


if __name__ == "__main__": main()
