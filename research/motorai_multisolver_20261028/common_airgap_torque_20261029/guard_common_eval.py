"""One-worker, retry0 owner guard for an explicitly released frozen evaluation.

No scheduler registration, training, FEA, foreign process termination, or retry.
Only a fresh runs directory is accepted. The CPU preflight is an existing pinned
receipt; this guard does not repeat checkpoint deserialization or create models.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

RUN_IDS = tuple(f"{arm}_s{seed}" for arm in ("A0", "A2", "C", "G") for seed in (17, 29, 43))


def require(condition, message):
    if not condition: raise ValueError(message)


def utc(): return dt.datetime.now(dt.timezone.utc)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(4 * 1024**2): digest.update(block)
    return digest.hexdigest()


def pinned_json(spec):
    path = Path(spec["path"]).resolve(strict=True)
    require(sha(path) == spec["sha256"], f"SHA mismatch: {path.name}")
    return json.loads(path.read_text(encoding="utf-8-sig"))


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)  # one attempt; no hidden retry


def validate_plan(plan):
    require(plan["status"] == "RELEASED_FROZEN_EVALUATION", "plan is HOLD")
    require(plan["workers"] == 1 and plan["retry_limit"] == 0 and plan["global_max_seconds"] == 7200
            and plan["per_model_max_seconds"] == 1800, "execution bounds changed")
    require(plan["training_updates"] == plan["new_FEA"] == plan["new_prior_solves"] == 0, "prohibited work in plan")
    require([row["id"] for row in plan["jobs"]] == list(RUN_IDS), "exact twelve-run fixed order required")
    require(len({row["job"]["path"] for row in plan["jobs"]}) == 12
            and len({row["out"] for row in plan["jobs"]}) == 12, "duplicate jobs/output paths")
    deadline = dt.datetime.fromisoformat(plan["deadline_UTC"])
    require(deadline.tzinfo is not None and deadline.utcoffset() == dt.timedelta(0), "deadline must be UTC")
    require(plan["disk_free_min_bytes"] >= 1024**3, "unbounded disk admission")
    root = Path(plan["runs_root"]).resolve()
    require(root.parent == Path(plan["stage_root"]).resolve() and root.name == "runs", "runs root outside stage")
    for row in plan["jobs"]:
        require(Path(row["out"]).resolve() == root / row["id"], "job output escapes owned runs root")
    return deadline


def parse_gpu_pids(output):
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if lines == ["No running processes found"]: return []
    require(all(line.isdigit() for line in lines), "GPU ownership probe returned unknown format")
    return sorted(set(map(int, lines)))


def gpu_compute_pids():
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                            capture_output=True, text=True, timeout=15, creationflags=flags, check=False)
    require(result.returncode == 0, f"GPU probe failed with code {result.returncode}")
    return parse_gpu_pids(result.stdout)


def owner_identity(process):
    return {"pid": int(process.pid), "create_time": float(process.create_time())}


def identity_matches(psutil, identity):
    try:
        return psutil.Process(identity["pid"]).create_time() == identity["create_time"]
    except psutil.NoSuchProcess:
        return False


def capture_owned_tree(psutil, root_identity):
    require(identity_matches(psutil, root_identity), "owned root PID identity changed")
    root = psutil.Process(root_identity["pid"])
    children = []
    for child in root.children(recursive=True):
        try: children.append(owner_identity(child))
        except psutil.NoSuchProcess: pass
    require(identity_matches(psutil, root_identity), "owned root changed while recording children")
    return [root_identity] + children


def terminate_owned(psutil, identities):
    """Terminate only recorded PID/create-time pairs; a recycled PID is skipped."""
    matched = []
    for identity in reversed(identities):
        if identity_matches(psutil, identity):
            process = psutil.Process(identity["pid"])
            try:
                if process.create_time() == identity["create_time"]:
                    process.terminate(); matched.append((identity, process))
            except psutil.NoSuchProcess: pass
    if matched:
        psutil.wait_procs([p for _, p in matched], timeout=5)
    for identity, process in matched:
        if identity_matches(psutil, identity):
            try:
                if process.is_running(): process.kill()
            except psutil.NoSuchProcess: pass


def limit_reason(now, deadline, elapsed_global, elapsed_model=None, stop=False):
    if stop: return "STOP_FILE"
    if now >= deadline: return "UTC_DEADLINE"
    if elapsed_global >= 7200: return "GLOBAL_7200_SECONDS"
    if elapsed_model is not None and elapsed_model >= 1800: return "MODEL_1800_SECONDS"
    return None


def worker_environment(base, pythonpath):
    environment = dict(base)
    # The CPU-only metadata preflight may intentionally hide CUDA. Its empty
    # mask must not leak into a separately approved GPU evaluation worker.
    if environment.get("CUDA_VISIBLE_DEVICES", "").strip() == "":
        environment.pop("CUDA_VISIBLE_DEVICES", None)
    environment.update(PYTHONPATH=pythonpath, PYTHONDONTWRITEBYTECODE="1", CUBLAS_WORKSPACE_CONFIG=":4096:8")
    return environment


def run_guard(plan, plan_sha):
    deadline = validate_plan(plan)
    require(Path(sys.executable).resolve() == Path(plan["python_executable"]).resolve(), "wrong pinned Python runtime")
    require(Path(__file__).resolve() == Path(plan["guard"]["path"]).resolve()
            and sha(__file__) == plan["guard"]["sha256"], "guard source not pinned")
    require(utc() < deadline, "deadline already expired")
    for source in plan["source_files"]:
        require(sha(source["path"]) == source["sha256"], "staged source SHA differs")
    preflight = pinned_json(plan["preflight_receipt"])
    require(preflight["status"] == "PASS" and preflight["scope"] == "CPU_METADATA_AND_SAVED_SOURCE_ONLY"
            and len(preflight["records"]) == 12 and {r["id"] for r in preflight["records"]} == set(RUN_IDS), "CPU twelve-checkpoint preflight missing")
    geometry = pinned_json(plan["geometry_manifest"])
    require(geometry["status"] == "ADMITTED" and len(geometry["records"]) == 32, "geometry admission missing")
    preflight_rows = {row["id"]: row for row in preflight["records"]}
    jobs = {}
    for row in plan["jobs"]:
        job = pinned_json(row["job"])
        require(job["id"] == row["id"] and job["release_status"] == "RELEASED_FROZEN_EVALUATION"
                and job["deadline_UTC"] == plan["deadline_UTC"], "job release/deadline mismatch")
        require(job["checkpoint"]["sha256"] == preflight_rows[row["id"]]["checkpoint_sha256"], "job checkpoint differs from CPU preflight")
        require(job["source_files"] == plan["source_files"] and job["geometry_manifest"] == plan["geometry_manifest"], "job source/geometry release differs")
        require(not Path(row["out"]).exists(), "job already exists; retry0")
        jobs[row["id"]] = job
    root = Path(plan["runs_root"])
    root.mkdir(parents=False, exist_ok=False)
    import psutil
    started = time.monotonic(); owner = owner_identity(psutil.Process(os.getpid()))
    status = {"status": "RUNNING", "plan_sha256": plan_sha, "owner": owner, "started_UTC": utc().isoformat(),
              "new_training_updates": 0, "new_FEA": 0, "new_prior_solves": 0, "automatic_retries": 0,
              "jobs": [{"id": run, "status": "NOT_STARTED"} for run in RUN_IDS]}
    atomic_json(root / "GUARD_OWNER.json", owner | {"plan_sha256": plan_sha})
    current = None; current_identities = []; known_children = {}
    def save():
        status["elapsed_seconds"] = time.monotonic() - started
        atomic_json(root / "BATCH_STATUS.json", status)
    def check(model_start=None):
        reason = limit_reason(utc(), deadline, time.monotonic() - started,
                              None if model_start is None else time.monotonic() - model_start,
                              any(Path(path).exists() for path in plan["stop_files"]))
        if reason: raise InterruptedError(reason)
    save()
    try:
        for index, row in enumerate(plan["jobs"]):
            check()
            require(shutil.disk_usage(root).free >= plan["disk_free_min_bytes"], "disk free-space admission failed")
            pids = gpu_compute_pids()
            require(not pids, f"GPU compute occupied before {row['id']}: {pids}; no foreign process will be stopped")
            require(not Path(row["out"]).exists(), "output appeared; refusing retry")
            job = jobs[row["id"]]
            require(sha(row["job"]["path"]) == row["job"]["sha256"], "job changed before launch")
            require(sha(plan["evaluator"]["path"]) == plan["evaluator"]["sha256"], "evaluator changed before launch")
            environment = worker_environment(os.environ, plan["pythonpath"])
            command = [plan["python_executable"], "-B", "-u", plan["evaluator"]["path"], "--job", row["job"]["path"],
                       "--job-sha256", row["job"]["sha256"], "--out", row["out"]]
            entry = status["jobs"][index]
            entry.update(status="STARTING", started_UTC=utc().isoformat(), job_sha256=row["job"]["sha256"])
            save()
            model_start = time.monotonic()
            log = root / (row["id"] + ".log")
            with log.open("xb") as handle:
                current = subprocess.Popen(command, cwd=plan["stage_root"], env=environment,
                                           stdout=handle, stderr=subprocess.STDOUT,
                                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                identity = owner_identity(psutil.Process(current.pid))
                entry.update(status="RUNNING", owner=identity)
                current_identities = [identity]; known_children = {}
                save()
                while current.poll() is None:
                    check(model_start)
                    try:
                        current_identities = capture_owned_tree(psutil, identity)
                        for child in current_identities:
                            known_children[(child["pid"], child["create_time"])] = child
                    except (psutil.NoSuchProcess, ValueError):
                        if current.poll() is None: raise
                    time.sleep(.5)
                returncode = current.wait()
            # An evaluator must not leave a live descendant, even after exit0.
            remaining = [child for child in known_children.values() if identity_matches(psutil, child)]
            if remaining:
                terminate_owned(psutil, remaining)
                raise RuntimeError("owned worker left running descendants; stopped only recorded owners")
            entry.update(returncode=returncode, elapsed_seconds=time.monotonic() - model_start, completed_UTC=utc().isoformat())
            current = None; current_identities = []; known_children = {}
            require(entry["elapsed_seconds"] <= plan["per_model_max_seconds"], "worker exceeded model wall bound before exit was observed")
            result_path = Path(row["out"]) / "RESULT.json"
            if result_path.exists():
                result = json.loads(result_path.read_text(encoding="utf-8"))
                entry.update(result_sha256=sha(result_path), evaluator_status=result.get("status"), graphs_processed=result.get("graphs_processed"))
            else:
                result = {}; entry["evaluator_status"] = "MISSING_RESULT"
            require(returncode == 0 and result.get("status") == "COMPLETED" and result.get("graphs_processed") == 1440
                    and result.get("checkpoint_sha256") == job["checkpoint"]["sha256"]
                    and result.get("job_sha256") == row["job"]["sha256"]
                    and result.get("reproduction_gate", {}).get("status") == "PASS_ARITHMETIC_ONLY", f"worker {row['id']} incomplete or failed")
            entry["status"] = "COMPLETED"
            atomic_json(root / (row["id"] + "_GUARD_RECEIPT.json"), entry)
            save()
        check()
        status.update(status="COMPLETED", completed_UTC=utc().isoformat(), completed_models=12,
                      completed_snapshots=17280, scientific_scope="diagnostics only; no new improvement gate")
    except BaseException as exc:
        status.update(status="STOPPED" if isinstance(exc, InterruptedError) else "INCOMPLETE_NO_CLAIM", error=repr(exc))
        for entry in status["jobs"]:
            if entry["status"] in ("STARTING", "RUNNING"):
                entry.update(status="INCOMPLETE_NO_CLAIM", error=repr(exc))
        raise
    finally:
        owners = {**known_children, **{(p["pid"], p["create_time"]): p for p in current_identities}}
        if owners:
            terminate_owned(psutil, list(owners.values()))
        if current is not None:
            try: current.wait(timeout=10)
            except subprocess.TimeoutExpired: status["owned_process_exit_unconfirmed"] = True
        status["completed_models"] = sum(entry["status"] == "COMPLETED" for entry in status["jobs"])
        status["completed_UTC"] = utc().isoformat()
        for entry in status["jobs"]:
            if "started_UTC" in entry:
                atomic_json(root / (entry["id"] + "_GUARD_RECEIPT.json"), entry)
        save()
        atomic_json(root / "GUARD_RECEIPT.json", {key: value for key, value in status.items() if key != "jobs"})
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    plan = pinned_json({"path": str(args.plan), "sha256": args.plan_sha256})
    run_guard(plan, args.plan_sha256)


if __name__ == "__main__": main()
