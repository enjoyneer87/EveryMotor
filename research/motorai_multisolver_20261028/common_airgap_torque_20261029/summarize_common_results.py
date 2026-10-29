"""JSON-only, fail-closed summary of the fixed 12-model common-airgap evaluation.

No torch/numpy imports, checkpoint loads, NPZ reads, network calls, or subprocesses.
Requires the frozen local package plus a collected results directory. The latter
contains runs/<id>/RESULT.json, runs/<id>_GUARD_RECEIPT.json, and runs/{BATCH_STATUS,
GUARD_RECEIPT}.json (the guard JSONs may instead be at the collection root).
COLLECTION_RECEIPT.json must list files with relative/path, sha256, and bytes.
Non-JSON collection members are explicitly left unopened. Every consumed JSON
result/receipt must be in that collection manifest. Output is a NEW file only.
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import statistics
import sys

ARMS = ("A0", "A2", "C", "G")
SEEDS = (17, 29, 43)
RUN_IDS = tuple(f"{arm}_s{seed}" for arm in ARMS for seed in SEEDS)
PHASES = {"all45": 45, "seen9": 9, "unused36": 36, "interior_unused": 32, "after_last_seen": 4}
FIELD_NUMBERS = ("vector_nrmse_percent", "magnitude_nrmse_percent", "vector_rmse_T", "magnitude_rmse_T")
TORQUE_NUMBERS = ("rmse_Nm", "truth_rms_Nm", "nrmse_percent", "pred_mean_Nm", "truth_mean_Nm",
                  "mean_error_rms_normalized_percent", "pred_ripple_Nm", "truth_ripple_Nm")
DEFAULT_PLAN_SHA = "0fcac205b8810f7ffece50b343158f187710d2cdb047b0026ea0c9c01c21cc24"
SCOPE = "E1-oldTRAIN-exposed innerdev32; saved-field common-operator diagnostics only"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def integer(value, name, minimum=0):
    require(type(value) is int and value >= minimum, f"invalid integer {name}")
    return value


def number(value, name, nullable=False):
    if value is None and nullable:
        return None
    require(type(value) in (int, float) and math.isfinite(value), f"nonfinite/non-numeric {name}")
    return float(value)


def digest_ok(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def reject_constant(value):
    raise ValueError(f"nonfinite JSON constant: {value}")


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path, digest=None, size=None):
    path = Path(path)
    require(path.suffix.lower() == ".json", "only JSON evidence may be opened")
    data = path.read_bytes()
    if digest is not None:
        require(digest_ok(digest) and sha(data) == digest, f"JSON SHA mismatch: {path.name}")
    if size is not None:
        require(len(data) == integer(size, "file bytes"), f"JSON byte count mismatch: {path.name}")
    value = json.loads(data.decode("utf-8-sig"), parse_constant=reject_constant, object_pairs_hook=unique_object)
    return value, {"sha256": sha(data), "bytes": len(data)}


def safe_relative(value):
    require(isinstance(value, str) and "\\" not in value, "collection path must use relative forward slashes")
    path = PurePosixPath(value)
    require(not path.is_absolute() and path.parts and all(p not in ("..", ".") and ":" not in p for p in path.parts), "unsafe collection path")
    return path.as_posix()


def inside(root, relative):
    result = (root / safe_relative(relative)).resolve()
    require(result.is_relative_to(root.resolve()), "path escapes evidence root")
    return result


def package_json(package, stage, spec):
    remote = spec["path"].replace("\\", "/")
    prefix = stage.rstrip("/") + "/"
    require(remote.startswith(prefix), "pinned JSON outside frozen package")
    path = inside(package, remote[len(prefix):])
    return read_json(path, spec["sha256"], spec.get("bytes"))[0]


def close(actual, expected, name):
    if expected is None:
        require(actual is None, f"{name}: undefined reference must remain null")
    else:
        actual = number(actual, name)
        require(math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-10), f"{name}: stored arithmetic differs")


def distribution(values):
    """Never silently reduce sample count for undefined metrics."""
    values = [number(v, "distribution", nullable=True) for v in values]
    valid = [v for v in values if v is not None]
    result = {"expected_count": len(values), "defined_count": len(valid), "undefined_count": len(values) - len(valid),
              "mean": None, "min": None, "max": None, "range": None, "median": None,
              "status": "DEFINED" if len(valid) == len(values) and valid else "UNDEFINED_NO_SILENT_DROP"}
    if result["status"] == "DEFINED":
        result.update(mean=statistics.fmean(valid), min=min(valid), max=max(valid),
                      range=max(valid) - min(valid), median=statistics.median(valid))
    return result


def torque_summary(samples, single_case=False):
    require(samples, "empty torque samples")
    pred = [number(r["pred_Nm"], "pred torque") for r in samples]
    truth = [number(r["truth_Nm"], "truth torque") for r in samples]
    rms = math.sqrt(math.fsum(t * t for t in truth) / len(truth))
    rmse = math.sqrt(math.fsum((p - t) ** 2 for p, t in zip(pred, truth)) / len(truth))
    require(rms > 0 or all(t == 0 for t in truth), "nonzero torque reference energy underflow")
    result = {"count": len(pred), "rmse_Nm": rmse, "truth_rms_Nm": rms,
              "nrmse_percent": 100 * rmse / rms if rms else None,
              "normalization_status": "DEFINED" if rms else "UNDEFINED_ZERO_REFERENCE"}
    if single_case:
        pm, tm = statistics.fmean(pred), statistics.fmean(truth)
        result.update(pred_mean_Nm=pm, truth_mean_Nm=tm,
                      mean_error_rms_normalized_percent=100 * abs(pm - tm) / rms if rms else None,
                      pred_ripple_Nm=max(pred) - min(pred), truth_ripple_Nm=max(truth) - min(truth))
    else:
        result["ripple"] = "NOT_POOLED_ACROSS_CASES"
    for key, value in result.items():
        if key not in ("normalization_status", "ripple"):
            number(value, key, nullable=True)
    return result


def check_numbers(stored, expected, context):
    for key, value in expected.items():
        if isinstance(value, str):
            require(stored.get(key) == value, f"{context}/{key}: status mismatch")
        else:
            close(stored.get(key), value, f"{context}/{key}")


def check_field_partitions(sufficient, cases, high_pa, a1_counts):
    statistic_keys = ("count", "weight_sum", "vector_sse", "magnitude_sse", "truth_energy")

    def equal_sum(total, parts, context):
        for key in statistic_keys:
            expected = math.fsum(row[key] for row in parts)
            if key == "count":
                require(total[key] == expected, f"{context}: field count sum differs")
            else:
                close(total[key], expected, f"{context}/{key}")

    for case in cases:
        for region in ("all", "a1"):
            all_count = sufficient[f"all45/case:{case}/{region}/area"]["count"]
            require(all_count % 45 == 0, "case all45 field count not divisible by 45")
            per_snapshot = all_count // 45
            if region == "a1" and a1_counts is not None:
                require(per_snapshot == a1_counts[case], "a1 count differs from admitted native element IDs")
            for phase, steps in PHASES.items():
                for weight in ("area", "unweighted"):
                    require(sufficient[f"{phase}/case:{case}/{region}/{weight}"]["count"] == per_snapshot * steps,
                            "case field phase count differs from fixed phase coverage")
    pa_groups = []
    if high_pa:
        pa_groups.append(("PA_ge_70", sorted(high_pa)))
    if set(cases) - high_pa:
        pa_groups.append(("PA_lt_70", sorted(set(cases) - high_pa)))
    for phase in PHASES:
        for region in ("all", "a1"):
            for weight in ("area", "unweighted"):
                key = f"{phase}/{region}/{weight}"
                equal_sum(sufficient[key], [sufficient[f"{phase}/case:{c}/{region}/{weight}"] for c in cases], key)
                equal_sum(sufficient[key], [sufficient[f"{phase}/{label}/{region}/{weight}"] for label, _ in pa_groups], key + "/PA partition")
                for label, selected in pa_groups:
                    pa_key = f"{phase}/{label}/{region}/{weight}"
                    equal_sum(sufficient[pa_key], [sufficient[f"{phase}/case:{c}/{region}/{weight}"] for c in selected], pa_key)
    group_suffixes = [""] + [f"/case:{c}" for c in cases] + [f"/{label}" for label, _ in pa_groups]
    for group in group_suffixes:
        for region in ("all", "a1"):
            for weight in ("area", "unweighted"):
                suffix = f"{group}/{region}/{weight}"
                for total, parts in (("all45", ("seen9", "unused36")), ("unused36", ("interior_unused", "after_last_seen"))):
                    equal_sum(sufficient[total + suffix], [sufficient[part + suffix] for part in parts], total + suffix + "/phase partition")


def check_field_metrics(result, cases, high_pa, a1_counts=None):
    expected_keys = set()
    for phase in PHASES:
        groups = [phase] + [f"{phase}/case:{case}" for case in cases]
        if high_pa:
            groups.append(f"{phase}/PA_ge_70")
        if set(cases) - high_pa:
            groups.append(f"{phase}/PA_lt_70")
        expected_keys.update(f"{group}/{region}/{weight}" for group in groups
                             for region in ("all", "a1") for weight in ("area", "unweighted"))
    fields, sufficient = result["field_metrics"], result["field_sufficient_statistics"]
    require(set(fields) == set(sufficient) == expected_keys, "missing/extra field scope or case (no silent drop)")
    truth_signature = {}
    for key in sorted(expected_keys):
        s = sufficient[key]
        count = integer(s["count"], f"{key}/count", 1)
        ws = number(s["weight_sum"], "weight_sum")
        ve, me, te = (number(s[n], n) for n in ("vector_sse", "magnitude_sse", "truth_energy"))
        require(ws > 0 and min(ve, me, te) >= 0, "negative sufficient statistics")
        expected = {"count": count, "vector_rmse_T": math.sqrt(ve / ws), "magnitude_rmse_T": math.sqrt(me / ws),
                    "vector_nrmse_percent": 100 * math.sqrt(ve / te) if te else None,
                    "magnitude_nrmse_percent": 100 * math.sqrt(me / te) if te else None,
                    "normalization_status": "DEFINED" if te else "UNDEFINED_ZERO_REFERENCE"}
        check_numbers(fields[key], expected, key)
        if key.endswith("/unweighted"):
            close(ws, count, f"{key}/unweighted count")
        truth_signature[key] = (count, ws, te)
    check_field_partitions(sufficient, cases, high_pa, a1_counts)
    return truth_signature


def check_result(result, job, job_pin, cases, high_pa, a1_counts=None):
    run = job["id"]
    require(result["status"] == "COMPLETED" and result["graphs_processed"] == 1440, f"{run}: evaluation incomplete")
    require(result["id"] == run and result["arm"] == job["arm"] and result["seed"] == job["seed"], "run identity mismatch")
    require(result["job_sha256"] == job_pin["sha256"] and result["checkpoint_sha256"] == job["checkpoint"]["sha256"], "result/checkpoint/job hash mismatch")
    require(result["phase_encoding"] == job["phase_encoding"], "phase encoding changed")
    for key, expected in (("selection_sha256", job["selection"]["sha256"]),
                          ("bundle_manifest_sha256", job["bundle_manifest_sha256"]),
                          ("geometry_manifest_sha256", job["geometry_manifest"]["sha256"])):
        require(result[key] == expected, f"{key} differs")
    require(all(result[key] == 0 for key in ("training_updates", "new_FEA", "new_prior_solves", "automatic_retries")), "forbidden work/retry recorded")
    require(result["verdict"] == "DIAGNOSTICS_ONLY_NO_NEW_IMPROVEMENT_GATE", "result scope changed")
    gate = result["reproduction_gate"]
    require(gate["status"] == "PASS_ARITHMETIC_ONLY" and gate["tolerance_pp"] == .0001, "reproduction gate changed")
    require(set(gate["absolute_differences_pp"]) == {"all45_original_eval_pp", "seen9_original_eval_pp", "seen9_training_anchor_pp"}, "missing reproduction anchor")
    require(all(0 <= number(v, "reproduction delta") < .0001 for v in gate["absolute_differences_pp"].values()), "reproduction anchor failed")
    truth_signature = check_field_metrics(result, cases, high_pa, a1_counts)
    rows = result["torque_samples"]
    pairs = [(integer(row["case"], "case"), integer(row["step"], "step")) for row in rows]
    require(len(rows) == len(set(pairs)) == 1440 and set(pairs) == {(c, s) for c in cases for s in range(45)}, "torque case/phase coverage mismatch")
    require(set(result["case_torque"]) == {str(c) for c in cases}, "per-case torque coverage mismatch")
    for case in cases:
        expected = torque_summary([r for r in rows if r["case"] == case], single_case=True)
        require(expected["count"] == 45, "per-case torque phase count mismatch")
        check_numbers(result["case_torque"][str(case)], expected, f"torque case {case}")
    check_numbers(result["pooled_torque"], torque_summary(rows), "pooled torque")
    case_mean = distribution([result["case_torque"][str(c)]["nrmse_percent"] for c in cases])["mean"]
    close(result["mean_case_torque_nrmse_percent"], case_mean, "mean of case torque nRMSE")
    predictions = result["prediction_files"]
    require(len(predictions) == 32 and {r["case_index"] for r in predictions} == set(cases), "prediction manifest coverage differs")
    for row in predictions:
        require(row["file"] == f"a1_case{row['case_index']:04d}.npz" and digest_ok(row["sha256"]), "invalid prediction metadata")
        integer(row["bytes"], "prediction bytes", 1)
    return {"fields": truth_signature, "torque": {f"{r['case']}:{r['step']}": r["truth_Nm"] for r in rows}}


def metrics_for_run(result, cases, high_pa):
    flat = {}
    for key, values in result["field_metrics"].items():
        if "/case:" not in key:
            for name in FIELD_NUMBERS:
                flat[f"field/{key}/{name}"] = values[name]
    case_rows = {}
    for case in cases:
        row = dict(result["case_torque"][str(case)])
        row.update(signed_mean_error_Nm=row["pred_mean_Nm"] - row["truth_mean_Nm"],
                   absolute_mean_error_Nm=abs(row["pred_mean_Nm"] - row["truth_mean_Nm"]),
                   signed_ripple_error_Nm=row["pred_ripple_Nm"] - row["truth_ripple_Nm"],
                   absolute_ripple_error_Nm=abs(row["pred_ripple_Nm"] - row["truth_ripple_Nm"]))
        case_rows[str(case)] = row
    grouped_torque = {}
    for label, selected in (("all45", set(cases)), ("PA_ge_70", high_pa), ("PA_lt_70", set(cases) - high_pa)):
        if not selected:
            grouped_torque[label] = {"status": "NO_REGISTERED_CASES", "case_ids": []}
            continue
        pooled = torque_summary([r for r in result["torque_samples"] if r["case"] in selected])
        distributions = {name: distribution([case_rows[str(c)][name] for c in sorted(selected)])
                         for name in TORQUE_NUMBERS + ("signed_mean_error_Nm", "absolute_mean_error_Nm", "signed_ripple_error_Nm", "absolute_ripple_error_Nm")}
        grouped_torque[label] = {"case_ids": sorted(selected), "pooled": pooled, "case_distributions": distributions,
                                "zero_reference_cases": [c for c in sorted(selected) if case_rows[str(c)]["normalization_status"] != "DEFINED"]}
        for name in ("rmse_Nm", "truth_rms_Nm", "nrmse_percent"):
            flat[f"torque/{label}/pooled/{name}"] = pooled[name]
        for name, dist in distributions.items():
            flat[f"torque/{label}/case_mean/{name}"] = dist["mean"]
    worst = {}
    for metric in ("nrmse_percent", "rmse_Nm", "absolute_mean_error_Nm", "absolute_ripple_error_Nm"):
        defined = [c for c in cases if case_rows[str(c)][metric] is not None]
        ordered = sorted(defined, key=lambda c: (-case_rows[str(c)][metric], c))
        worst[metric] = {"ranked_cases": [{"case": c, "value": case_rows[str(c)][metric]} for c in ordered],
                         "undefined_cases": [c for c in cases if c not in defined],
                         "expected_case_count": len(cases)}
    return {"metrics": flat, "torque_groups": grouped_torque, "case_torque": case_rows, "worst_torque": worst,
            "field_normalization_status": {k: v["normalization_status"] for k, v in result["field_metrics"].items()},
            "per_case_all45_field": {str(c): {f"{region}/{weight}": result["field_metrics"][f"all45/case:{c}/{region}/{weight}"]
                                               for region in ("all", "a1") for weight in ("area", "unweighted")} for c in cases}}


def aggregate_runs(runs):
    require(set(runs) == set(RUN_IDS), "arm aggregation requires all twelve runs")
    keys = set(runs[RUN_IDS[0]]["metrics"])
    require(all(set(r["metrics"]) == keys for r in runs.values()), "run metric sets differ")
    arms = {arm: {key: distribution([runs[f"{arm}_s{s}"]["metrics"][key] for s in SEEDS]) for key in sorted(keys)} for arm in ARMS}
    paired = {}
    for left, right in (("A0", "A2"), ("C", "G"), ("A0", "C"), ("A0", "G")):
        comparison = {}
        for key in sorted(keys):
            rows = []
            for seed in SEEDS:
                a, b = runs[f"{left}_s{seed}"]["metrics"][key], runs[f"{right}_s{seed}"]["metrics"][key]
                rows.append({"seed": seed, "left": a, "right": b, "delta_left_minus_right": None if a is None or b is None else a - b})
            comparison[key] = {"paired": rows, "delta_summary": distribution([r["delta_left_minus_right"] for r in rows]),
                               "units": "percentage_points" if "percent" in key else ("T" if key.endswith("_T") else "Nm")}
        paired[f"{left}_minus_{right}"] = comparison
    return arms, paired


def collection_inventory(root, plan_sha):
    receipt, pin = read_json(root / "COLLECTION_RECEIPT.json")
    require(receipt.get("status") == "VERIFIED" and receipt.get("terminal_status") == "COMPLETED", "collection not completed/verified")
    require(receipt["guard_plan_sha256"] == plan_sha and receipt["raw_or_checkpoint_files_transferred"] == 0
            and receipt["automatic_retries"] == 0 and receipt["hash_size_failures"] == 0, "collection plan/raw/hash/retry scope differs")
    for key in ("remote_snapshot_before", "remote_snapshot_after"):
        snapshot = receipt[key]
        require(snapshot["status"] == "TERMINAL_READY" and snapshot["terminal_status"] == "COMPLETED"
                and snapshot["guard_plan_sha256"] == plan_sha and snapshot["live_owned"] == []
                and snapshot["binary_payloads_opened_or_hashed"] == 0, "collection taken before owned processes stopped or outside JSON/text scope")
    files = receipt["files"]
    require(isinstance(files, list) and files, "empty collection manifest")
    by_relative = {}
    verified, unopened = [], []
    for row in files:
        relative = row.get("relative", row.get("path"))
        relative = safe_relative(relative)
        require(relative not in by_relative and relative != "COLLECTION_RECEIPT.json", "duplicate/self collection member")
        require(digest_ok(row["sha256"]), "invalid collection SHA")
        integer(row["bytes"], "collection bytes")
        by_relative[relative] = row
        if Path(relative).suffix.lower() == ".json":
            read_json(inside(root, relative), row["sha256"], row["bytes"])
            verified.append(relative)
        else:
            unopened.append(relative)
    return by_relative, {"receipt": pin, "declared_file_count": len(files), "json_hash_size_verified_count": len(verified),
                         "unopened_non_json_count": len(unopened), "unopened_non_json_paths": unopened,
                         "status": "PASS_LISTED_JSON_HASH_SIZE_ONLY", "literal_64_file_assumption": False}


def collected_json(root, inventory, relative):
    relative = safe_relative(relative)
    require(relative in inventory, f"required evidence absent from collection manifest: {relative}")
    row = inventory[relative]
    return read_json(inside(root, relative), row["sha256"], row["bytes"])


def receipt_location(root, inventory, name):
    found = [p for p in (f"runs/{name}", name) if p in inventory]
    require(len(found) == 1, f"missing/ambiguous receipt {name}")
    return found[0]


def check_owner(owner):
    integer(owner["pid"], "owner PID", 1)
    require(number(owner["create_time"], "owner create_time") > 0, "invalid owner identity")


def check_batch(batch, guard, plan, plan_sha):
    for name, row in (("batch", batch), ("guard", guard)):
        require(row["status"] == "COMPLETED" and row["completed_models"] == 12 and row["completed_snapshots"] == 17280, f"{name}: incomplete twelve-model batch")
        require(row["plan_sha256"] == plan_sha, "guard plan SHA differs")
        require(all(row[k] == 0 for k in ("new_training_updates", "new_FEA", "new_prior_solves", "automatic_retries")), "forbidden batch operation")
        require(0 <= number(row["elapsed_seconds"], "batch elapsed") <= 7200, "batch wall bound exceeded")
        completed = dt.datetime.fromisoformat(row["completed_UTC"])
        deadline = dt.datetime.fromisoformat(plan["deadline_UTC"])
        require(completed.tzinfo is not None and completed <= deadline, "batch completion beyond UTC deadline")
        require(not row.get("owned_process_exit_unconfirmed", False), "owned process exit unconfirmed")
        check_owner(row["owner"])
    require(batch["owner"] == guard["owner"], "guard owner identity differs")
    require([r["id"] for r in batch["jobs"]] == list(RUN_IDS), "batch fixed run list differs")


def summarize(input_dir, package_dir, plan_sha=DEFAULT_PLAN_SHA):
    output = {"schema_version": 1, "status": "INCOMPLETE_NO_CLAIM", "scope": SCOPE,
              "existing_verdicts": {"A2": "NO_IMPROVEMENT_CLAIM", "C_vs_G": "NO_IMPROVEMENT_CLAIM"},
              "verdict_policy": "UNCHANGED; diagnostic reporting cannot upgrade preregistered outcomes",
              "uncertainty_policy": "three-seed arithmetic ranges only; no CI, p-value, or significance claim",
              "operations": {"NPZ_opened": 0, "checkpoints_loaded": 0, "models_created": 0,
                             "training_updates": 0, "new_evaluation_runs": 0, "new_FEA": 0, "remote_calls": 0},
              "run_checks": {run: {"status": "NOT_CHECKED"} for run in RUN_IDS}, "errors": []}
    root, package = Path(input_dir).resolve(), Path(package_dir).resolve()
    try:
        plan, plan_pin = read_json(package / "BATCH_PLAN.json", plan_sha)
        require(plan["status"] == "RELEASED_FROZEN_EVALUATION" and [r["id"] for r in plan["jobs"]] == list(RUN_IDS), "wrong frozen batch plan")
        require(plan["workers"] == 1 and plan["retry_limit"] == 0 and plan["global_max_seconds"] == 7200 and plan["per_model_max_seconds"] == 1800, "plan resource bounds changed")
        stage = plan["stage_root"]
        geometry = package_json(package, stage, plan["geometry_receipt_copy"])
        require(plan["geometry_receipt_copy"]["sha256"] == plan["geometry_manifest"]["sha256"], "geometry copy hash differs")
        require(geometry["status"] == "ADMITTED" and len(geometry["records"]) == 32, "geometry incomplete")
        jobs = {row["id"]: package_json(package, stage, row["job"]) for row in plan["jobs"]}
        first = jobs[RUN_IDS[0]]
        selection = package_json(package, stage, first["selection"])
        dev = {integer(r["case_index"], "development case"): r for r in selection["records"] if r["split"] == "development"}
        require(len(dev) == sum(r["split"] == "development" for r in selection["records"]) == 32, "development selection coverage differs")
        cases = sorted(dev)
        high_pa = {c for c in cases if number(dev[c]["native_input"]["PhaseAdvance"], "PhaseAdvance") >= 70}
        require({r["case_index"] for r in geometry["records"]} == set(cases), "geometry/selection cases differ")
        a1_counts = {}
        for row in geometry["records"]:
            require(row["status"] == "ADMITTED" and row["steps_checked"] == 45 and row["source_sha256"] == dev[row["case_index"]]["source_sha256"], "geometry/source binding differs")
            ids = [integer(value, "admitted a1 ID") for value in row["a1_native_element_ids"]]
            require(ids and len(ids) == len(set(ids)), "admitted a1 IDs empty or duplicated")
            a1_counts[row["case_index"]] = len(ids)
        require(geometry["selection_sha256"] == first["selection"]["sha256"] and geometry["bundle_manifest_sha256"] == first["bundle_manifest_sha256"], "geometry/common bundle differs")
        preflight = package_json(package, stage, plan["preflight_receipt"])
        require(preflight["status"] == "PASS" and len(preflight["records"]) == 12 and {r["id"] for r in preflight["records"]} == set(RUN_IDS), "preflight incomplete")
        checkpoints = {r["id"]: r["checkpoint_sha256"] for r in preflight["records"]}
        require(len(set(checkpoints.values())) == 12, "checkpoint identities not unique")
        collection, output["collection"] = collection_inventory(root, plan_sha)
        batch, batch_pin = collected_json(root, collection, receipt_location(root, collection, "BATCH_STATUS.json"))
        guard, guard_pin = collected_json(root, collection, receipt_location(root, collection, "GUARD_RECEIPT.json"))
        check_batch(batch, guard, plan, plan_sha)
        batch_rows = {r["id"]: r for r in batch["jobs"]}
        run_summaries, reference_truth = {}, None
        for row in plan["jobs"]:
            run, job = row["id"], jobs[row["id"]]
            try:
                arm, seed = run.split("_s")
                require(job["id"] == run and job["arm"] == arm and job["seed"] == int(seed), "job identity differs")
                require(job["source_files"] == plan["source_files"] and job["geometry_manifest"] == plan["geometry_manifest"], "source/geometry differs between models")
                require(job["selection"] == first["selection"] and job["bundle_manifest_sha256"] == first["bundle_manifest_sha256"], "selection/bundle differs between models")
                require(job["checkpoint"]["sha256"] == checkpoints[run], "checkpoint differs from preflight")
                require(job["deadline_UTC"] == plan["deadline_UTC"] and job["phase_encoding"] == ("electrical" if arm == "A2" else "mechanical"), "job deadline/encoding differs")
                result, result_pin = collected_json(root, collection, f"runs/{run}/RESULT.json")
                receipt, receipt_pin = collected_json(root, collection, receipt_location(root, collection, f"{run}_GUARD_RECEIPT.json"))
                require(receipt == batch_rows[run], "per-run receipt differs from batch entry")
                require(receipt["status"] == "COMPLETED" and receipt["returncode"] == 0 and receipt["graphs_processed"] == 1440, "worker receipt incomplete")
                require(receipt["result_sha256"] == result_pin["sha256"] and receipt["job_sha256"] == row["job"]["sha256"], "worker receipt hash differs")
                require(0 <= number(receipt["elapsed_seconds"], "worker elapsed") <= 1800, "worker bound exceeded")
                check_owner(receipt["owner"])
                truth = check_result(result, job, row["job"], cases, high_pa, a1_counts)
                original_eval = package_json(package, stage, job["original_eval"])
                original_train = package_json(package, stage, job["original_train"])
                require(original_eval["status"] == "COMPLETED" and original_eval["graphs_processed"] == 1440
                        and original_eval["checkpoint_sha256"] == job["checkpoint"]["sha256"], "original evaluation binding differs")
                anchors = {
                    "all45_original_eval_pp": (result["field_metrics"]["all45/all/area"]["vector_nrmse_percent"], original_eval["metrics"]["all45/all"]["B_vector_nRMSE_percent"]),
                    "seen9_original_eval_pp": (result["field_metrics"]["seen9/all/area"]["vector_nrmse_percent"], original_eval["metrics"]["seen9/all"]["B_vector_nRMSE_percent"]),
                    "seen9_training_anchor_pp": (result["field_metrics"]["seen9/all/area"]["vector_nrmse_percent"], original_train["metrics"][-1]["development"]["all"]["vector_nRMSE_percent"]),
                }
                for name, (actual, expected) in anchors.items():
                    delta = abs(number(actual, name) - number(expected, name))
                    require(delta < .0001, "recomputed original-score reproduction failed")
                    close(result["reproduction_gate"]["absolute_differences_pp"][name], delta, name)
                if reference_truth is None:
                    reference_truth = truth
                else:
                    require(truth["torque"] == reference_truth["torque"], "truth torque differs between models")
                    require(set(truth["fields"]) == set(reference_truth["fields"]), "truth field scopes differ")
                    for key, vals in truth["fields"].items():
                        for i, value in enumerate(vals):
                            close(value, reference_truth["fields"][key][i], f"common truth {key}/{i}")
                run_summaries[run] = metrics_for_run(result, cases, high_pa)
                run_summaries[run].update(checkpoint_sha256=result["checkpoint_sha256"], result=result_pin,
                                          guard_receipt=receipt_pin, reproduction_gate=result["reproduction_gate"],
                                          prediction_npz_metadata_count=len(result["prediction_files"]))
                output["run_checks"][run] = {"status": "PASS", "graphs": 1440, "cases": 32, "phases_per_case": 45}
            except (ValueError, KeyError, TypeError, OSError, OverflowError) as exc:
                output["run_checks"][run] = {"status": "FAIL", "error": str(exc)}
                output["errors"].append(f"{run}: {exc}")
        require(not output["errors"] and len(run_summaries) == 12, "all twelve runs must pass before aggregation")
        arms, paired = aggregate_runs(run_summaries)
        output.update(status="COMPLETED_DIAGNOSTICS_ONLY", completed_models=12, completed_graphs=17280,
                      cases=cases, high_PA_case_ids=sorted(high_pa), runs=run_summaries, arms=arms, paired=paired,
                      provenance={"batch_plan": plan_pin, "batch_status": batch_pin, "guard_receipt": guard_pin,
                                  "selection_sha256": first["selection"]["sha256"], "bundle_manifest_sha256": first["bundle_manifest_sha256"],
                                  "geometry_manifest_sha256": plan["geometry_manifest"]["sha256"],
                                  "frozen_source_pins": plan["source_files"], "source_verification": "IDENTICAL_HASH_PIN_DECLARATIONS_IN_FROZEN_PLAN_AND_ALL12_JOBS; source bytes not reopened by JSON-only summary",
                                  "native_source_hashes": {str(c): dev[c]["source_sha256"] for c in cases}})
    except (ValueError, KeyError, TypeError, OSError, OverflowError) as exc:
        output["errors"].append(str(exc))
    output["passed_run_count"] = sum(v["status"] == "PASS" for v in output["run_checks"].values())
    return output


def self_test():
    """Pure in-memory tests: no evidence files, models, or output files touched."""
    assert distribution([1, 2, 3])["range"] == 2
    assert distribution([1, None, 3])["mean"] is None
    zero = torque_summary([{"pred_Nm": 1, "truth_Nm": 0}] * 45, True)
    assert zero["nrmse_percent"] is None and zero["rmse_Nm"] == 1
    assert zero["normalization_status"] == "UNDEFINED_ZERO_REFERENCE"
    rows = [{"pred_Nm": 2, "truth_Nm": 1}] * 45
    assert torque_summary(rows, True)["nrmse_percent"] == 100
    assert torque_summary(rows)["ripple"] == "NOT_POOLED_ACROSS_CASES"
    runs = {run: {"metrics": {"field/test/vector_nrmse_percent": float(i)}} for i, run in enumerate(RUN_IDS)}
    arms, paired = aggregate_runs(runs)
    assert arms["A0"]["field/test/vector_nrmse_percent"]["mean"] == 1
    assert paired["A0_minus_A2"]["field/test/vector_nrmse_percent"]["delta_summary"]["mean"] == -3
    for invalid in ("../bad.json", "/bad.json", "C:/bad.json", "a\\b.json"):
        try:
            safe_relative(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("unsafe path accepted")
    for action in (lambda: number(float("nan"), "nan"), lambda: integer(True, "bool"),
                   lambda: unique_object([("a", 1), ("a", 2)]), lambda: close(0, None, "zero reference"),
                   lambda: aggregate_runs({k: v for k, v in runs.items() if k != "G_s43"})):
        try:
            action()
        except ValueError:
            pass
        else:
            raise AssertionError("invalid evidence accepted")
    # Complete 32-case x 45-phase synthetic result, including every field scope.
    cases, high = list(range(32)), set(range(16, 32))
    job = {"id": "A0_s17", "arm": "A0", "seed": 17, "phase_encoding": "mechanical",
           "checkpoint": {"sha256": "a" * 64}, "selection": {"sha256": "b" * 64},
           "bundle_manifest_sha256": "c" * 64, "geometry_manifest": {"sha256": "d" * 64}}
    pin = {"sha256": "e" * 64}
    result = {"id": "A0_s17", "arm": "A0", "seed": 17, "phase_encoding": "mechanical",
              "status": "COMPLETED", "graphs_processed": 1440, "job_sha256": "e" * 64,
              "checkpoint_sha256": "a" * 64, "selection_sha256": "b" * 64,
              "bundle_manifest_sha256": "c" * 64, "geometry_manifest_sha256": "d" * 64,
              "training_updates": 0, "new_FEA": 0, "new_prior_solves": 0, "automatic_retries": 0,
              "verdict": "DIAGNOSTICS_ONLY_NO_NEW_IMPROVEMENT_GATE", "field_metrics": {}, "field_sufficient_statistics": {},
              "reproduction_gate": {"status": "PASS_ARITHMETIC_ONLY", "tolerance_pp": .0001,
                                    "absolute_differences_pp": {k: 0. for k in ("all45_original_eval_pp", "seen9_original_eval_pp", "seen9_training_anchor_pp")}}}
    for phase, steps in PHASES.items():
        groups = [(phase, 32), (f"{phase}/PA_ge_70", 16), (f"{phase}/PA_lt_70", 16)]
        groups += [(f"{phase}/case:{case}", 1) for case in cases]
        for group, ncases in groups:
            for region, elements in (("all", 3), ("a1", 1)):
                for weighting, weight in (("area", 2), ("unweighted", 1)):
                    key = f"{group}/{region}/{weighting}"
                    count = steps * ncases * elements
                    result["field_sufficient_statistics"][key] = {"count": count, "weight_sum": count * weight,
                                                                  "vector_sse": count * weight * .25,
                                                                  "magnitude_sse": count * weight * .25, "truth_energy": count * weight}
                    result["field_metrics"][key] = {"count": count, "vector_nrmse_percent": 50., "magnitude_nrmse_percent": 50.,
                                                     "vector_rmse_T": .5, "magnitude_rmse_T": .5, "normalization_status": "DEFINED"}
    result["torque_samples"] = [{"case": c, "step": s, "pred_Nm": 2., "truth_Nm": 1.} for c in cases for s in range(45)]
    result["case_torque"] = {str(c): torque_summary([r for r in result["torque_samples"] if r["case"] == c], True) for c in cases}
    result["pooled_torque"] = torque_summary(result["torque_samples"])
    result["mean_case_torque_nrmse_percent"] = 100.
    result["prediction_files"] = [{"case_index": c, "file": f"a1_case{c:04d}.npz", "sha256": "f" * 64, "bytes": 1} for c in cases]
    check_result(result, job, pin, cases, high, {c: 1 for c in cases})
    extracted = metrics_for_run(result, cases, high)
    assert extracted["torque_groups"]["PA_ge_70"]["pooled"]["count"] == 16 * 45
    assert len(extracted["worst_torque"]["nrmse_percent"]["ranked_cases"]) == 32
    mutations = (
        lambda r: r["torque_samples"].pop(),
        lambda r: r["torque_samples"][0].update(step=1),
        lambda r: r["field_metrics"].pop("all45/a1/area"),
        lambda r: r["case_torque"].pop("0"),
        lambda r: r["pooled_torque"].update(nrmse_percent=99.),
        lambda r: r.update(checkpoint_sha256="0" * 64),
        lambda r: r["field_metrics"]["all45/a1/area"].update(vector_nrmse_percent=float("nan")),
        lambda r: (r["field_sufficient_statistics"]["seen9/case:0/a1/area"].update(count=10),
                   r["field_metrics"]["seen9/case:0/a1/area"].update(count=10)),
        lambda r: (r["field_sufficient_statistics"]["all45/case:0/a1/area"].update(vector_sse=90.),
                   r["field_metrics"]["all45/case:0/a1/area"].update(vector_nrmse_percent=100., vector_rmse_T=1.)),
    )
    for mutation in mutations:
        invalid = copy.deepcopy(result)
        mutation(invalid)
        try:
            check_result(invalid, job, pin, cases, high, {c: 1 for c in cases})
        except (ValueError, KeyError):
            pass
        else:
            raise AssertionError("incomplete/tampered synthetic result accepted")
    print("PASS: in-memory full-schema/arithmetic/coverage/hash/undefined/path tests; file writes=0")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--package-dir", type=Path)
    parser.add_argument("--plan-sha256", default=DEFAULT_PLAN_SHA)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    parser.error("--input-dir, --package-dir and --out are required") if any(v is None for v in (args.input_dir, args.package_dir, args.out)) else None
    require(args.out.suffix.lower() == ".json", "summary output must be JSON")
    require(not args.out.exists(), "refusing to overwrite an existing summary")
    result = summarize(args.input_dir, args.package_dir, args.plan_sha256)
    with args.out.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "passed_run_count": result["passed_run_count"], "errors": result["errors"], "out": str(args.out)}))
    return 0 if result["status"] == "COMPLETED_DIAGNOSTICS_ONLY" else 2


if __name__ == "__main__":
    sys.exit(main())
