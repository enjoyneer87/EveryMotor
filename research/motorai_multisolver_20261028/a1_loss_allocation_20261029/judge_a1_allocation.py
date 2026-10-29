"""JSON-only, fail-closed preregistered judgment for all six L0/L05 runs.
No model, checkpoint, NPZ, remote query, retry, or adaptive threshold path.
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

ARMS = ("L0", "L05")
SEEDS = (17, 29, 43)
RUN_IDS = tuple(f"{arm}_s{seed}" for arm in ARMS for seed in SEEDS)
PHASES = {"all45": 45, "seen9": 9, "unused36": 36, "interior_unused": 32, "after_last_seen": 4}
FIELD_NUMBERS = ("vector_nrmse_percent", "magnitude_nrmse_percent", "vector_rmse_T", "magnitude_rmse_T")
TORQUE_NUMBERS = ("rmse_Nm", "truth_rms_Nm", "nrmse_percent", "pred_mean_Nm", "truth_mean_Nm",
                  "mean_error_rms_normalized_percent", "pred_ripple_Nm", "truth_ripple_Nm")
PROPOSAL_SHA = "65da672079d62a44d3436d3d018050be525de7072e5eb7c766ea615b58d196b7"
STAGE = "C:/work/EveryMotor_A1Loss_20261029"
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


def check_base_fields(result, cases, high_pa, a1_counts=None):
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


PRIMARY = "field/all45/a1/area/vector_nrmse_percent"
TORQUE = "torque/all45/pooled/nrmse_percent"
NO_HARM = {
    "all45_all_Bvector": ("field/all45/all/area/vector_nrmse_percent", .5),
    "seen9_all_Bvector": ("field/seen9/all/area/vector_nrmse_percent", .5),
    "copper_Bvector": ("field/all45/copper/area/vector_nrmse_percent", 1.),
    "PA_ge70_mean_case_Bvector": ("PA_mean_case/all45/PA_ge_70/all/mean_case_Bvector_nrmse_percent", 1.),
    "common_Arkkio_pooled_torque_nRMSE": (TORQUE, 1.),
}
PAIRED_ORDER = tuple(f"{arm}_s{seed}" for seed in SEEDS for arm in ARMS)


def check_proposal(proposal):
    require(proposal["arms"] == {"L0": {"alpha": 0.}, "L05": {"alpha": .5}}, "proposal arms differ")
    fixed = proposal["fixed"]
    require(fixed["seeds"] == list(SEEDS) and fixed["updates_per_run"] == 88565 and fixed["full_runs"] == 6
            and fixed["width"] == 32 and fixed["depth"] == 3 and fixed["angle"] == "mechanical", "proposal fixed design differs")
    rule = proposal["proposed_decision_rule_to_freeze_before_release"]
    require(rule["mean_no_harm_L05_minus_L0_pp"] == {key: value[1] for key, value in NO_HARM.items()}, "proposal no-harm margins differ")
    require(rule["primary_improvement"] == "delta>=1.0pp AND delta>2*spread", "proposal primary gate differs")
    require(rule["torque_improvement_claim"] == "Requires its own delta>=1pp AND delta>2*torque_spread; primary a1 success alone does not imply torque success", "proposal torque gate differs")


class Evidence:
    """Resolve recorded JSON copies by exact SHA; runtime outputs must be collected."""
    def __init__(self, root, package, inventory):
        self.root, self.package, self.inventory = root, package, inventory
        self.static_candidates = None

    def result(self, relative):
        return collected_json(self.root, self.inventory, relative)

    def pinned(self, spec):
        require(digest_ok(spec["sha256"]), "invalid JSON pin")
        remote = spec["path"].replace("\\", "/")
        prefix = STAGE + "/"
        if remote.startswith(prefix):
            relative = safe_relative(remote[len(prefix):])
            if relative.startswith("runs/"):
                value, actual = self.result(relative)
                require(actual["sha256"] == spec["sha256"] and ("bytes" not in spec or actual["bytes"] == spec["bytes"]), "runtime JSON pin differs")
                return value
            path = inside(self.package, relative)
            if path.exists():
                return read_json(path, spec["sha256"], spec.get("bytes"))[0]
        # Previously admitted geometry/selection may have a different remote root.
        # Only a byte-identical JSON copy inside this frozen package is accepted.
        if self.static_candidates is None:
            self.static_candidates = sorted(self.package.rglob("*.json"))
        for path in self.static_candidates:
            value, actual = read_json(path)
            if actual["sha256"] == spec["sha256"]:
                require("bytes" not in spec or actual["bytes"] == spec["bytes"], "static JSON size differs")
                return value
        raise ValueError("missing pinned static JSON: " + Path(remote).name)


def check_all_fields(result, cases, high_pa, a1_counts):
    base = {key: {k: v for k, v in result[key].items() if "/copper/" not in k}
            for key in ("field_metrics", "field_sufficient_statistics")}
    signature = check_base_fields(base, cases, high_pa, a1_counts)
    fields, stats = result["field_metrics"], result["field_sufficient_statistics"]
    expected = {f"{phase}{group}/copper/{weight}" for phase in PHASES
                for group in [""] + [f"/case:{case}" for case in cases] for weight in ("area", "unweighted")}
    require({k for k in fields if "/copper/" in k} == {k for k in stats if "/copper/" in k} == expected, "copper scope coverage differs")
    for key in expected:
        s = stats[key]; count = integer(s["count"], "copper field count", 1)
        ws, ve, me, te = (number(s[n], n) for n in ("weight_sum", "vector_sse", "magnitude_sse", "truth_energy"))
        require(ws > 0 and min(ve, me, te) >= 0, "invalid copper sufficient statistics")
        expected_values = dict(count=count, vector_rmse_T=math.sqrt(ve/ws), magnitude_rmse_T=math.sqrt(me/ws),
            vector_nrmse_percent=100*math.sqrt(ve/te) if te else None, magnitude_nrmse_percent=100*math.sqrt(me/te) if te else None,
            normalization_status="DEFINED" if te else "UNDEFINED_ZERO_REFERENCE")
        check_numbers(fields[key], expected_values, key)
        if key.endswith("/unweighted"): close(ws, count, key + "/count")
        signature[key] = (count, ws, te)
    def sums(total, parts):
        for name in ("count", "weight_sum", "vector_sse", "magnitude_sse", "truth_energy"):
            close(stats[total][name], math.fsum(stats[p][name] for p in parts), total + "/" + name)
    for case in cases:
        count = stats[f"all45/case:{case}/copper/area"]["count"]
        require(count % 45 == 0, "copper snapshot count differs")
        for phase, steps in PHASES.items():
            for weight in ("area", "unweighted"):
                require(stats[f"{phase}/case:{case}/copper/{weight}"]["count"] == count//45*steps, "copper phase counts differ")
    for phase in PHASES:
        for weight in ("area", "unweighted"):
            sums(f"{phase}/copper/{weight}", [f"{phase}/case:{c}/copper/{weight}" for c in cases])
    for group in [""] + [f"/case:{c}" for c in cases]:
        for weight in ("area", "unweighted"):
            suffix = f"{group}/copper/{weight}"
            sums("all45"+suffix, ["seen9"+suffix, "unused36"+suffix])
            sums("unused36"+suffix, ["interior_unused"+suffix, "after_last_seen"+suffix])
    return signature


def check_copper_B2(result, cases):
    rows = result["conductor_B2"]
    require(rows and {r["case"] for r in rows} == set(cases), "copper conductor cases differ")
    keys = [(r["case"], r["native_reg_code"], r["phase_scope"]) for r in rows]
    require(len(set(keys)) == len(rows), "duplicate conductor phase")
    case_codes = {(c, code) for c, code, phase in keys}
    require(set(keys) == {(c, code, phase) for c, code in case_codes for phase in PHASES}, "conductor phase coverage differs")
    totals = {phase: [0., 0., 0.] for phase in PHASES}; truth = {}
    for row in rows:
        phase = row["phase_scope"]
        require(row["snapshots"] == PHASES[phase], "conductor phase count differs")
        area, p, t, pm, tm = (number(row[k], k) for k in ("mean_area_m2", "pred_spatial_mean_B2_T2", "true_spatial_mean_B2_T2", "pred_square_spatial_mean_B_T2", "true_square_spatial_mean_B_T2"))
        require(area > 0 and min(p, t, pm, tm) >= 0 and pm <= p+1e-9 and tm <= t+1e-9, "invalid conductor B2")
        close(row["B2_signed_relative_percent"], 100*(p-t)/t if t else None, "conductor relative bias")
        for i, value in enumerate((area*abs(p-t), area*(p-t), area*t)): totals[phase][i] += value
        truth[str((row["case"], row["native_reg_code"], phase))] = (area, t, tm)
    require(set(result["copper_B2"]) == set(PHASES), "B2 phase coverage differs")
    for phase, (absolute, signed, reference) in totals.items():
        row = result["copper_B2"][phase]
        expected = dict(area_weighted_relative_MAE_percent=100*absolute/reference if reference else None,
            signed_bias_percent=100*signed/reference if reference else None,
            weighted_absolute_error_T2_m2=absolute, weighted_signed_error_T2_m2=signed, weighted_reference_T2_m2=reference,
            normalization_status="DEFINED" if reference else "UNDEFINED_ZERO_REFERENCE")
        check_numbers(row, expected, "copper B2/"+phase)
    return truth


def check_new_result(result, job, job_pin, train, cases, high_pa, a1_counts):
    require(result["status"] == "COMPLETED" and result["graphs_processed"] == 1440, "evaluation incomplete")
    require(result["id"] == job["id"] and result["arm"] == job["arm"] and result["seed"] == job["seed"]
            and result["alpha"] == job["alpha"] and result["phase_encoding"] == "mechanical", "evaluation identity differs")
    require(result["job_sha256"] == job_pin["sha256"] and result["checkpoint_sha256"] == job["checkpoint"]["sha256"]
            and train["checkpoint_sha256"] == job["checkpoint"]["sha256"], "checkpoint/job chain differs")
    for key, expected in (("selection_sha256", job["selection"]["sha256"]), ("bundle_manifest_sha256", job["bundle_manifest_sha256"]),
                          ("geometry_manifest_sha256", job["geometry_manifest"]["sha256"]),
                          ("training_mask_manifest_sha256", job["training_mask_manifest"]["sha256"]),
                          ("original_train_sha256", job["original_train"]["sha256"])):
        require(result[key] == expected, "evaluation provenance differs: "+key)
    require(all(result[k] == 0 for k in ("training_updates", "new_FEA", "new_prior_solves", "automatic_retries")), "evaluation performed undeclared work")
    require(result["verdict"] == "PENDING_ALL_SIX_PREREGISTERED_JUDGMENT", "premature/incompatible model verdict")
    gate = result["reproduction_gate"]
    require(gate["status"] == "PASS_ARITHMETIC_ONLY" and gate["tolerance_pp"] == .0001
            and set(gate["absolute_differences_pp"]) == {"seen9_training_anchor_pp"}, "reproduction contract differs")
    difference = abs(number(result["field_metrics"]["seen9/all/area"]["vector_nrmse_percent"], "seen9")
                     - number(train["metrics"][-1]["development"]["all"]["vector_nRMSE_percent"], "train anchor"))
    require(difference < .0001, "training anchor failed")
    close(gate["absolute_differences_pp"]["seen9_training_anchor_pp"], difference, "anchor recomputation")
    signature = check_all_fields(result, cases, high_pa, a1_counts)
    pa = result["PA_mean_case_metrics"]
    expected_pa = {f"{phase}/{group}/{region}" for phase in PHASES for group in ("PA_ge_70", "PA_lt_70") for region in ("all", "a1", "copper")}
    require(set(pa) == expected_pa, "PA mean-case coverage differs")
    for phase in PHASES:
        for label, selected in (("PA_ge_70", high_pa), ("PA_lt_70", set(cases)-high_pa)):
            for region in ("all", "a1", "copper"):
                key = f"{phase}/{label}/{region}"; row = pa[key]
                require(row["case_ids"] == sorted(selected) and row["case_count"] == len(selected), "PA case selection differs")
                values = [result["field_metrics"][f"{phase}/case:{c}/{region}/area"]["vector_nrmse_percent"] for c in sorted(selected)]
                expected = distribution(values)["mean"]
                close(row["mean_case_Bvector_nrmse_percent"], expected, "PA mean-case arithmetic")
                require(row["normalization_status"] == ("DEFINED" if expected is not None else "UNDEFINED_NO_SILENT_DROP"), "PA undefined status differs")
    copper_truth = check_copper_B2(result, cases)
    expected_pairs = {(c, step) for c in cases for step in range(45)}
    samples = result["torque_samples"]
    pairs = [(integer(r["case"], "case"), integer(r["step"], "step")) for r in samples]
    require(len(samples) == len(set(pairs)) == 1440 and set(pairs) == expected_pairs, "torque phase coverage differs")
    require(set(result["case_torque"]) == {str(c) for c in cases}, "torque case coverage differs")
    for case in cases: check_numbers(result["case_torque"][str(case)], torque_summary([r for r in samples if r["case"] == case], True), "case torque")
    check_numbers(result["pooled_torque"], torque_summary(samples), "pooled torque")
    close(result["mean_case_torque_nrmse_percent"], distribution([result["case_torque"][str(c)]["nrmse_percent"] for c in cases])["mean"], "mean case torque")
    snapshots = result["snapshot_errors"]; snapshot_pairs = [(r["case"], r["step"]) for r in snapshots]
    require(len(snapshots) == len(set(snapshot_pairs)) == 1440 and set(snapshot_pairs) == expected_pairs, "snapshot error coverage differs")
    for row in snapshots:
        require(row["phase_scope"] == ("seen9" if row["step"] % 5 == 0 else "unused36"), "snapshot phase label differs")
        for key in ("vector_nrmse_percent", "magnitude_nrmse_percent"): number(row[key], key, nullable=True)
    predictions = result["prediction_files"]
    require(len(predictions) == 32 and {r["case_index"] for r in predictions} == set(cases), "prediction metadata coverage differs")
    for row in predictions:
        require(row["file"] == f"a1_case{row['case_index']:04d}.npz" and digest_ok(row["sha256"]), "prediction metadata differs")
        integer(row["bytes"], "prediction bytes", 1)
    return dict(fields=signature, copper=copper_truth, torque={str((r["case"],r["step"])):r["truth_Nm"] for r in samples})


def metrics_for_new_run(result, cases, high_pa):
    value = metrics_for_run(result, cases, high_pa)
    flat = value["metrics"]
    for key, row in result["PA_mean_case_metrics"].items():
        flat["PA_mean_case/"+key+"/mean_case_Bvector_nrmse_percent"] = row["mean_case_Bvector_nrmse_percent"]
    for phase, row in result["copper_B2"].items():
        for metric in ("signed_bias_percent", "area_weighted_relative_MAE_percent"):
            flat[f"copper_B2/{phase}/{metric}"] = row[metric]
    selectors = {"all45":lambda s:True, "seen9":lambda s:s%5==0, "unused36":lambda s:s%5!=0,
                 "interior_unused":lambda s:s%5!=0 and s<=40, "after_last_seen":lambda s:s>40}
    for phase, selector in selectors.items():
        rows = [r for r in result["snapshot_errors"] if selector(r["step"])]
        for metric in ("vector_nrmse_percent", "magnitude_nrmse_percent"):
            flat[f"snapshot_mean/{phase}/{metric}"] = distribution([r[metric] for r in rows])["mean"]
    value.update(copper_B2=result["copper_B2"], PA_mean_case_metrics=result["PA_mean_case_metrics"],
                 historical_A0_reference=result["historical_A0_reference"], metadata_hashes=result["metadata_hashes"],
                 checkpoint_sha256=result["checkpoint_sha256"])
    return value


def decide(runs):
    require(set(runs) == set(RUN_IDS), "no verdict until all six runs pass")
    keys = set(runs[RUN_IDS[0]]["metrics"])
    require(all(set(r["metrics"]) == keys for r in runs.values()), "model metric sets differ")
    arms = {arm:{key:distribution([runs[f"{arm}_s{s}"]["metrics"][key] for s in SEEDS]) for key in sorted(keys)} for arm in ARMS}
    paired = {}
    for key in sorted(keys):
        rows = []
        for seed in SEEDS:
            left, right = (runs[f"{a}_s{seed}"]["metrics"][key] for a in ARMS)
            rows.append(dict(seed=seed, L0=left, L05=right, delta_L0_minus_L05=None if left is None or right is None else left-right))
        paired[key] = {"paired":rows,"delta_summary":distribution([r["delta_L0_minus_L05"] for r in rows])}
    required = {PRIMARY, TORQUE} | {key for key, _ in NO_HARM.values()}
    require(required <= keys, "decision metric missing")
    require(all(arms[a][key]["status"] == "DEFINED" for a in ARMS for key in required), "undefined preregistered metric; no silent drop")
    def gate(key):
        delta = arms["L0"][key]["mean"]-arms["L05"][key]["mean"]
        spread = max(arms[a][key]["range"] for a in ARMS)
        return dict(delta_L0_minus_L05_pp=delta, spread_pp=spread, twice_spread_pp=2*spread,
                    minimum_improvement_pass=delta>=1., spread_pass=delta>2*spread, pass_=delta>=1. and delta>2*spread)
    primary, torque = gate(PRIMARY), gate(TORQUE)
    no_harm = {}
    for name, (key, limit) in NO_HARM.items():
        change = arms["L05"][key]["mean"]-arms["L0"][key]["mean"]
        no_harm[name] = dict(change_L05_minus_L0_pp=change, limit_pp=limit, pass_=change<=limit)
    all_no_harm = all(v["pass_"] for v in no_harm.values())
    verdict = "IMPROVEMENT_CLAIM_CRITERIA_MET" if primary["pass_"] and all_no_harm else "NO_IMPROVEMENT_CLAIM"
    return dict(verdict=verdict, primary=primary, no_harm=no_harm, no_harm_pass=all_no_harm,
                torque_metric_improvement_gate=torque, torque_gate_is_separate=True,
                range_interpretation="descriptive three-seed max-min; not CI, significance, or equivalence"), arms, paired


def utc_value(value):
    parsed = dt.datetime.fromisoformat(value)
    require(parsed.tzinfo is not None and parsed.utcoffset() == dt.timedelta(0), "timestamp is not UTC")
    return parsed


def source_map(rows):
    require(isinstance(rows, list) and rows, "source declarations missing")
    result = {}
    for row in rows:
        name = PurePosixPath(row["path"].replace("\\", "/")).name
        require(name not in result and digest_ok(row["sha256"]), "duplicate/invalid source pin")
        integer(row["bytes"], "source bytes", 1)
        result[name] = row["sha256"]
    return result


def check_chain_plan(plan):
    require(plan["status"] == "RELEASED_ENGINEERING_THEN_FIXED_SIX_RUNS", "chain not released")
    exact = dict(workers=1, retry_limit=0, global_max_seconds=57600, engineering_max_seconds=900,
                 engineering_updates=2957, train_max_seconds=7200, eval_max_seconds=1800,
                 projection_safety_factor=1.25, collection_reserve_seconds=300)
    for key, value in exact.items(): require(number(plan[key], key) == value, "chain bound differs: "+key)
    require(plan["run_order"] == list(PAIRED_ORDER), "fixed paired order differs")
    require(plan["stage_root"].replace("\\", "/").rstrip("/") == STAGE
            and plan["runs_root"].replace("\\", "/").rstrip("/") == STAGE+"/runs", "owned stage differs")
    require(integer(plan["disk_free_min_bytes"], "disk minimum") >= 10*1024**3, "disk admission below proposal")
    require(STAGE+"/STOP" in [p.replace("\\", "/") for p in plan["stop_files"]], "stage STOP absent")
    utc_value(plan["deadline_UTC"])
    sources = source_map(plan["source_files"])
    for role, name in (("trainer", "train_a1_allocation.py"), ("evaluator", "evaluate_a1_allocation.py"), ("guard", "run_a1_chain.py")):
        require(plan[role]["sha256"] == sources[name], "role/source pin differs: "+role)
    require("dual_solver_mgn.py" in sources and "a1_allocation_loss.py" in sources, "training sources missing")
    return sources


def check_static(evidence, plan, sources, proposal):
    train, engineering, cpu, masks, template = (evidence.pinned(plan[k]) for k in
        ("training_plan", "engineering_plan", "cpu_admission", "mask_manifest", "evaluation_template"))
    require(cpu["status"] == "PASS" and masks["status"] == "ADMITTED", "CPU/native admission incomplete")
    require(set(cpu["initial_state_sha256_by_seed"]) == {str(s) for s in SEEDS}
            and all(digest_ok(v) for v in cpu["initial_state_sha256_by_seed"].values()), "CPU initial state coverage differs")
    for updates, expected_seeds in (("2957", {"17"}), ("88565", {"17", "29", "43"})):
        rows = cpu["expected_data_order_LR_trace_sha256"][updates]
        require(expected_seeds <= set(rows) <= {"17", "29", "43"} and all(digest_ok(v) for v in rows.values()), "CPU order/LR coverage differs")
    cpu_sources = source_map(cpu["source_files"])
    require({"dual_solver_mgn.py", "a1_allocation_loss.py", "train_a1_allocation.py"} <= set(cpu_sources), "CPU training source pins missing")
    require(all(sources.get(k) == v for k, v in cpu_sources.items()), "CPU/release source differs")
    for value, status in ((train, "RELEASED_FIXED_SIX_RUNS"), (engineering, "RELEASED_ENGINEERING_ONLY")):
        require(value["status"] == status and value["arms"] == proposal["arms"] and value["fixed_seeds"] == list(SEEDS)
                and value["seen_steps"] == list(range(0, 45, 5)), "training release/design differs")
        require(value["source_files"] == plan["source_files"] and value["guard_deadline_UTC"] == plan["deadline_UTC"], "training source/deadline differs")
        require(value["initial_state_sha256_by_seed"] == cpu["initial_state_sha256_by_seed"], "training CPU initialization differs")
        require(value["selection"]["sha256"] == masks["selection_sha256"] == proposal["fixed"]["selection_sha256"]
                and value["bundle_manifest_sha256"] == masks["bundle_manifest_sha256"]
                and value["mask_manifest_sha256"] == plan["mask_manifest"]["sha256"], "training data pins differ")
        spec = value["training"]
        for key, wanted in dict(width=32, depth=3, expected_optimizer_updates=88565, updates_per_A0_epoch=2952, smoke_updates=5).items():
            require(spec[key] == wanted, "fixed training value differs: "+key)
        require(spec["learning_rate"]["start"] == .001 and spec["learning_rate"]["end"] == .00001
                and spec["evaluation_updates"] == {"0": 5, "1": 2957, "5": 14765, "10": 29525, "20": 59045, "30": 88565}, "LR/assessment schedule differs")
        require(spec["retry"] == 0 and spec["wall_seconds_max"] <= 7200 and spec["torch_allocated_soft_limit_GiB"] <= 10, "training resource bound differs")
    require(train["selection"] == engineering["selection"] and train["bundle_manifest_sha256"] == engineering["bundle_manifest_sha256"], "engineering/full inputs differ")
    require(template["source_files"] == plan["source_files"] and template["deadline_UTC"] == plan["deadline_UTC"]
            and template["selection"] == train["selection"] and template["training_mask_manifest"] == plan["mask_manifest"], "evaluation template pins differ")
    require(template["release_status"] == "RELEASED_FROZEN_EVALUATION" and template["max_seconds"] == 1800
            and template["workers"] == 1 and template["retry_limit"] == 0 and template["phase_encoding"] == "mechanical", "evaluation release/bounds differ")
    require(all(template[k] == 0 for k in ("training_updates", "new_FEA", "new_prior_solves")), "evaluation scope differs")
    selection = evidence.pinned(train["selection"])
    geometry = evidence.pinned(template["geometry_manifest"])
    records = selection["records"]
    require(len(records) == 360 and len({r["case_index"] for r in records}) == 360, "selection coverage differs")
    require({r["split"] for r in records} == {"train", "development"}, "unexpected split")
    dev = {r["case_index"]: r for r in records if r["split"] == "development"}
    training = {r["case_index"]: r for r in records if r["split"] == "train"}
    require(len(dev) == 32 and len(training) == 328, "fixed split counts differ")
    cases = sorted(dev); high_pa = {c for c in cases if number(dev[c]["native_input"]["PhaseAdvance"], "PhaseAdvance") >= 70}
    require(high_pa and set(cases)-high_pa, "PA gate groups unavailable")
    require(geometry["status"] == "ADMITTED" and len(geometry["records"]) == 32
            and {r["case_index"] for r in geometry["records"]} == set(cases), "dev geometry incomplete")
    require(geometry["selection_sha256"] == train["selection"]["sha256"]
            and geometry["bundle_manifest_sha256"] == template["bundle_manifest_sha256"]
            and masks["dev_geometry_manifest_sha256"] == template["geometry_manifest"]["sha256"]
            and masks["common_dev_bundle_sha256"] == template["bundle_manifest_sha256"], "common mask/operator binding differs")
    a1_counts = {}
    for row in geometry["records"]:
        require(row["status"] == "ADMITTED" and row["steps_checked"] == 45 and row["source_sha256"] == dev[row["case_index"]]["source_sha256"], "dev geometry provenance differs")
        ids = [integer(i, "a1 ID") for i in row["a1_native_element_ids"]]
        require(ids and len(ids) == len(set(ids)), "a1 IDs empty/duplicated")
        a1_counts[row["case_index"]] = len(ids)
    require(masks["train_steps"] == list(range(0, 45, 5)) and masks["train_graphs"] == 2952 and masks["development_graphs"] == 288, "mask phase coverage differs")
    for key, selected in (("records", training), ("development_records", dev)):
        rows = masks[key]
        require(len(rows) == len(selected) and {r["case_index"] for r in rows} == set(selected), "mask case coverage differs")
        for row in rows:
            require(row["status"] == "ADMITTED" and row["source_sha256"] == selected[row["case_index"]]["source_sha256"]
                    and digest_ok(row["operator_sha256"]), "mask/source binding differs")
            require(len(row["graphs"]) == 9 and {g["step"] for g in row["graphs"]} == set(range(0,45,5)), "mask phase coverage differs")
            require(row["a1_native_element_ids"] and len(row["a1_native_element_ids"]) == len(set(row["a1_native_element_ids"])), "mask native IDs invalid")
            for graph in row["graphs"]:
                require(all(digest_ok(graph[k]) for k in ("sha256", "native_element_id_sha256", "a1_mask_sha256")), "mask graph hash invalid")
                require(0 < graph["a1_count"] <= graph["all_count"] and graph["a1_count"] == len(row["a1_native_element_ids"]), "mask node count differs")
                require(0 < number(graph["area_a1_m2"], "a1 area") <= number(graph["area_all_m2"], "all area"), "mask areas invalid")
                close(graph["q"], graph["area_a1_m2"]/graph["area_all_m2"], "mask q")
                close(graph["effective_a1_allocation_alpha05"], .5+.5*graph["q"], "effective loss allocation")
    return train, engineering, cpu, template, selection, cases, high_pa, a1_counts


def expected_metadata(plan, training, cpu, sources, seed, engineering=False):
    return dict(model_sha256=sources["dual_solver_mgn.py"], trainer_sha256=plan["trainer"]["sha256"],
        loss_module_sha256=sources["a1_allocation_loss.py"], plan_sha256=plan["engineering_plan" if engineering else "training_plan"]["sha256"],
        selection_sha256=training["selection"]["sha256"], bundle_manifest_sha256=training["bundle_manifest_sha256"],
        mask_manifest_sha256=plan["mask_manifest"]["sha256"], initial_tensor_state_sha256=cpu["initial_state_sha256_by_seed"][str(seed)],
        data_order_LR_trace_sha256=cpu["expected_data_order_LR_trace_sha256"]["2957" if engineering else "88565"][str(seed)])


def check_training(result, entry, expected, selection, engineering=False):
    updates = 2957 if engineering else 88565
    require(result["status"] == ("SMOKE_COMPLETED" if engineering else "COMPLETED") and result["training_updates"] == updates, "training incomplete")
    for key in ("arm", "seed", "alpha"): require(result[key] == entry[key], "training identity differs: "+key)
    require(result["parameters"] == 70182 and result["field_path_parameters"] == 30594 and result["phase_encoding"] == "mechanical"
            and result["data_arm"] == "A0" and result["old_checkpoint_reused"] is False, "architecture/angle/checkpoint reuse differs")
    for key in ("new_FEA", "new_prior_solves", "original_val_test_used"): require(result[key] == 0, "training scope differs: "+key)
    for key, wanted in expected.items(): require(digest_ok(wanted) and result.get(key) == wanted, "training provenance differs: "+key)
    require(digest_ok(result["checkpoint_sha256"]) and result["checkpoint_sha256"] == entry["checkpoint_sha256"], "checkpoint completion receipt differs")
    require(result["train_steps"] == list(range(0,45,5)) and result["train_graphs"] == 2952
            and result["train_conditions"] == 328 and result["development_conditions"] == 32, "training sample contract differs")
    for split, key in (("train", "train_groups"), ("development", "development_groups")):
        groups = sorted({r["geometry_group"] for r in selection["records"] if r["split"] == split})
        require(result[key] == groups, "training geometry groups differ")
    require(result["GPU_smoke"] == "PASS" and 0 <= number(result["peak_torch_allocated_bytes"], "GPU memory") <= 10*1024**3, "GPU engineering admission failed")
    require(result["max_updates"] == (2957 if engineering else None), "partial/full run differs")
    epochs = [0,1] if engineering else [0,1,5,10,20,30]
    require([m["epoch"] for m in result["metrics"]] == epochs and result["metrics"][-1]["training_updates"] == updates, "final assessment schedule differs")
    require(result["metrics"][-1]["development"]["all"]["snapshots"] == 288, "final anchor incomplete")
    for row in result["metrics"]:
        require(number(row["assessment_wall_seconds"], "assessment wall") > 0
                and number(row["development"]["all"]["vector_nRMSE_percent"], "anchor") >= 0, "invalid assessment")
    close(result["assessment_wall_seconds_total"], sum(m["assessment_wall_seconds"] for m in result["metrics"]), "assessment time total")


def check_receipt(entry, receipt, run_id, kind, sources, deadline):
    require(receipt == entry, "status and per-worker receipt differ")
    arm, seed = run_id.split("_s")
    ident = "engineering_"+run_id if kind == "engineering" else run_id+"_"+kind
    expected_out = STAGE+"/runs/"+("engineering/"+run_id if kind == "engineering" else run_id+"/"+kind)
    require(entry["id"] == ident and entry["kind"] == kind and entry["run_id"] == run_id and entry["arm"] == arm
            and entry["seed"] == int(seed) and entry["alpha"] == (0. if arm == "L0" else .5), "worker identity differs")
    require(entry["status"] == "COMPLETED" and entry["returncode"] == 0
            and entry["worker_status"] == ("SMOKE_COMPLETED" if kind == "engineering" else "COMPLETED"), "worker not successfully completed")
    require(entry["out"].replace("\\", "/") == expected_out and entry["source_sha256_by_name"] == sources, "worker source/output differs")
    check_owner(entry["owner"])
    start, end = utc_value(entry["started_UTC"]), utc_value(entry["completed_UTC"])
    require(start <= end <= deadline, "worker timestamp beyond frozen deadline")
    bound = dict(engineering=900, train=7200, eval=1800)[kind]
    require(0 <= number(entry["elapsed_seconds"], "worker elapsed") <= bound, "worker time cap exceeded")
    require(digest_ok(entry["result_sha256"]) and digest_ok(entry["checkpoint_sha256"]), "worker final pins invalid")
    return start, end


def compare_truth(left, right):
    require(left.keys() == right.keys(), "common reference families differ")
    for family in left:
        require(left[family].keys() == right[family].keys(), "common truth scope differs")
        for key, expected in left[family].items():
            actual = right[family][key]
            if isinstance(expected, (tuple, list)):
                require(len(actual) == len(expected), "reference statistic shape differs")
                for a, b in zip(actual, expected): close(a, b, "common truth "+family)
            else: close(actual, expected, "common truth "+family)


def judge(input_dir, package_dir, plan_path, plan_sha, proposal_path):
    output = dict(status="INCOMPLETE_NO_CLAIM", verdict="INCOMPLETE_NO_CLAIM", scope=SCOPE,
        graph_disconnection_contribution="UNKNOWN; this loss ablation does not measure native connectivity effect",
        operations=dict(JSON_only=True, NPZ_opened=0, checkpoints_loaded=0, model_imports=0,
                        training_updates=0, new_evaluation_runs=0, new_FEA=0, remote_calls=0),
        run_checks={run:{"status":"NOT_CHECKED"} for run in RUN_IDS}, errors=[])
    try:
        require(digest_ok(plan_sha), "externally frozen chain plan SHA required")
        plan, plan_pin = read_json(Path(plan_path), plan_sha)
        proposal, proposal_pin = read_json(Path(proposal_path), PROPOSAL_SHA); check_proposal(proposal)
        sources = check_chain_plan(plan)
        root, package = Path(input_dir).resolve(), Path(package_dir).resolve()
        inventory, collection = collection_inventory(root, plan_sha)
        evidence = Evidence(root, package, inventory)
        train_plan, engineering_plan, cpu, template, selection, cases, high_pa, a1_counts = check_static(evidence, plan, sources, proposal)
        status, status_pin = evidence.result("runs/CHAIN_STATUS.json")
        receipt, receipt_pin = evidence.result("runs/CHAIN_RECEIPT.json")
        owner, _ = evidence.result("runs/GUARD_OWNER.json")
        require(receipt == {k:v for k,v in status.items() if k != "jobs"}, "chain status/receipt differs")
        require(status["status"] == "COMPLETED" and status["chain_plan_sha256"] == status["plan_sha256"] == plan_sha, "chain incomplete/plan mismatch")
        require(status["scientific_verdict"] == "PENDING_PREREGISTERED_JSON_JUDGMENT", "chain premature verdict")
        for key, expected in dict(completed_train_runs=6, completed_eval_runs=6, completed_engineering_runs=2,
            full_training_updates=531390, engineering_updates=5914, evaluation_graphs=8640,
            new_FEA=0, new_prior_solves=0, automatic_retries=0).items(): require(status[key] == expected, "chain total differs: "+key)
        require(not status.get("owned_process_exit_unconfirmed", False), "owned worker exit unconfirmed")
        check_owner(status["owner"])
        require(owner == status["owner"] | {"chain_plan_sha256":plan_sha}, "guard owner differs")
        deadline = utc_value(plan["deadline_UTC"]); started, ended = utc_value(status["started_UTC"]), utc_value(status["completed_UTC"])
        require(started <= ended <= deadline and 0 <= number(status["elapsed_seconds"], "chain elapsed") <= 57600, "chain bounds exceeded")
        projection, _ = evidence.result("runs/ENGINEERING_PROJECTION_RECEIPT.json")
        require(status["engineering_projection"] == projection and projection["status"] == "PASS_RESOURCE_PROJECTION_ONLY", "engineering projection not passed")
        require(set(projection["arms"]) == set(ARMS) and projection["evaluation_reserve_seconds"] == 10800
                and projection["collection_reserve_seconds"] == 300, "projection scope differs")
        for row in projection["arms"].values():
            require(row["safety_factor"] == 1.25 and 0 < number(row["projected_train_seconds"], "projected train") <= 7200, "projection per-run bound failed")
        close(projection["projected_remaining_seconds"], 3*sum(r["projected_train_seconds"] for r in projection["arms"].values())+10800+300, "projection total")
        require(projection["projected_remaining_seconds"] <= projection["available_remaining_seconds"] <= 57600, "projection total bound failed")
        expected_jobs = [("L0_s17", "engineering"), ("L05_s17", "engineering")]+[(run, kind) for run in PAIRED_ORDER for kind in ("train", "eval")]
        require(len(status["jobs"]) == 14, "fixed14 job count differs")
        trains, runs, common_truth, previous_end = {}, {}, None, started
        for entry, (run, kind) in zip(status["jobs"], expected_jobs):
            worker_receipt, _ = evidence.result("runs/"+entry["id"]+"_GUARD_RECEIPT.json")
            job_start, job_end = check_receipt(entry, worker_receipt, run, kind, sources, deadline)
            require(previous_end <= job_start <= job_end <= ended, "workers overlap or order differs"); previous_end = job_end
            relative = "runs/"+("engineering/"+run if kind == "engineering" else run+"/"+kind)+"/RESULT.json"
            result, result_pin = evidence.result(relative)
            require(result_pin["sha256"] == entry["result_sha256"], "worker result completion hash differs")
            expected = expected_metadata(plan, train_plan, cpu, sources, entry["seed"], kind == "engineering")
            if kind != "eval":
                require(entry["plan_sha256"] == expected["plan_sha256"], "worker training plan differs")
                check_training(result, entry, expected, selection, kind == "engineering")
                require(entry["training_updates"] == result["training_updates"]
                        and all(entry[k] == expected[k] for k in ("initial_tensor_state_sha256", "data_order_LR_trace_sha256")), "worker initialization/order receipt differs")
                if kind == "train": trains[run] = (result, result_pin)
            else:
                job, job_pin = evidence.result("runs/"+run+"/eval_job.json")
                require(job_pin["sha256"] == entry["job_sha256"] and job["metadata_hashes"] == expected, "evaluation job expected metadata differs")
                fixed = copy.deepcopy(job)
                for key in ("id", "arm", "seed", "alpha", "checkpoint", "original_train", "metadata_hashes"): fixed.pop(key, None)
                template_fixed = copy.deepcopy(template)
                for key in ("id", "arm", "seed", "alpha", "checkpoint", "original_train", "metadata_hashes"): template_fixed.pop(key, None)
                require(fixed == template_fixed, "dynamic evaluation changed frozen template")
                require(job["id"] == run and job["arm"] == entry["arm"] and job["seed"] == entry["seed"] and job["alpha"] == entry["alpha"], "evaluation identity differs")
                trained, train_pin = trains[run]
                require(job["original_train"]["sha256"] == train_pin["sha256"] and job["original_train"]["bytes"] == train_pin["bytes"], "evaluation anchor pin differs")
                evidence.pinned(job["original_train"])
                require(result["metadata_hashes"] == expected and result["model_sha256"] == expected["model_sha256"]
                        and result["evaluator_sha256"] == plan["evaluator"]["sha256"], "evaluator/source metadata differs")
                require(entry["graphs_processed"] == 1440 and entry["checkpoint_sha256"] == trained["checkpoint_sha256"], "evaluation completion checkpoint differs")
                truth = check_new_result(result, job, job_pin, trained, cases, high_pa, a1_counts)
                if common_truth is None: common_truth = truth
                else: compare_truth(common_truth, truth)
                runs[run] = metrics_for_new_run(result, cases, high_pa)
                output["run_checks"][run] = dict(status="PASS", training_updates=88565, evaluation_graphs=1440,
                    training_result=train_pin, evaluation_result=result_pin, job=job_pin)
        first, second = status["jobs"][:2]
        require((utc_value(second["completed_UTC"])-utc_value(first["started_UTC"])).total_seconds() <= 900, "combined engineering exceeds900s")
        for seed in SEEDS:
            left, right = (trains[f"{arm}_s{seed}"][0] for arm in ARMS)
            require(all(left[key] == right[key] for key in ("initial_tensor_state_sha256", "data_order_LR_trace_sha256")), "paired initialization/order differs")
        decision, arms, paired = decide(runs)
        output.update(status="COMPLETE", verdict=decision["verdict"], decision=decision, runs=runs, arms=arms, paired=paired,
            cases=cases, high_PA_case_ids=sorted(high_pa), source_sha256_by_name=sources,
            evidence=dict(chain_plan=plan_pin, proposal=proposal_pin, chain_status=status_pin, chain_receipt=receipt_pin,
                          collection=collection, cpu_admission=plan["cpu_admission"], mask_manifest=plan["mask_manifest"]),
            verification_scope="JSON SHA/size, declared frozen source/checkpoint hashes and completion receipts; no binary payload reopened",
            exclusions=["independent final test", "AC loss qualification", "physical torque qualification", "native connectivity causal contribution"])
    except (ValueError, KeyError, TypeError, OSError, ArithmeticError) as exc:
        output["errors"].append(type(exc).__name__+": "+str(exc))
    output["passed_run_count"] = sum(r["status"] == "PASS" for r in output["run_checks"].values())
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--proposal", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    require(not args.out.exists(), "output already exists; retain previous judgment")
    result = judge(args.input, args.package, args.plan, args.plan_sha256, args.proposal)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False); stream.write("\n")
    print(json.dumps({k:result[k] for k in ("status", "verdict", "passed_run_count", "errors")}, ensure_ascii=False))
    return 0 if result["status"] == "COMPLETE" else 2


if __name__ == "__main__":
    sys.exit(main())


