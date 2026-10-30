"""Bounded, zero-update common a1/Arkkio evaluator for twelve frozen checkpoints.

This file is prepared/tested with synthetic data only until a separate job is
hash-frozen and released. No training/optimizer/prior/FEA path exists. Source,
geometry admission, checkpoints and graph bundles must all be explicitly pinned.
Never reuses a completed output directory; failures retain partial receipts.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import datetime as dt
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

from strict_airgap import (
    align_airgap_field, build_strict_airgap, field_sufficient_statistics,
    finish_statistics, merge_statistics, strict_arkkio_torque,
    summarize_torque45, validate_snapshot_coverage,
)

SEEN = frozenset(range(0, 45, 5))
ARMS = {"A0": "mechanical", "A2": "electrical", "C": "mechanical", "G": "mechanical"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path, check=lambda: None):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(4 * 1024**2):
            check()
            digest.update(block)
    return digest.hexdigest()


def pinned_path(spec, check=lambda: None):
    path = Path(spec["path"]).resolve(strict=True)
    require(path.is_file() and sha256(path, check) == spec["sha256"], f"SHA mismatch: {path.name}")
    if "bytes" in spec:
        require(path.stat().st_size == spec["bytes"], f"size mismatch: {path.name}")
    return path


def pinned_json(spec, check=lambda: None):
    return json.loads(pinned_path(spec, check).read_text(encoding="utf-8-sig"))


def child_file(root, relative):
    root = Path(root).resolve(strict=True)
    path = (root / relative).resolve(strict=True)
    require(path.is_relative_to(root), "artifact escapes declared root")
    return path


def integer_ids(value, name):
    array = np.asarray(value)
    require(array.ndim == 1 and array.dtype.kind in "iu" and len(array) > 0, f"invalid {name}")
    require(len(np.unique(array)) == len(array), f"duplicate {name}")
    require(array.dtype.kind != "u" or array.max() <= np.iinfo(np.int64).max, f"{name} exceeds int64")
    return array.astype(np.int64)


def exact_positions(have, need):
    order = np.argsort(have)
    positions = np.searchsorted(have[order], need)
    require(np.all(positions < len(have)), "missing native ID")
    require(np.array_equal(have[order][positions], need), "unregistered native ID")
    return order[positions]


def freeze(array):
    array = np.ascontiguousarray(array)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def load_operator(path):
    """Recompute a1 geometry from the admitted, hash-bound sidecar; no raw HDF5."""
    with np.load(path, allow_pickle=False) as z:
        values = {key: z[key] for key in z.files}
    admitted = np.sort(integer_ids(values["admitted_element_ids"], "admitted IDs"))
    a1_ids = integer_ids(values["a1_element_ids"], "a1 IDs")
    require(np.array_equal(a1_ids, np.sort(a1_ids)), "a1 sidecar IDs must be sorted")
    exact_positions(admitted, a1_ids)
    codes = integer_ids(values["region_codes"], "a1 region codes")
    require(values["symmetry_multiplier"].shape == () and values["symmetry_multiplier"].dtype.kind in "iu", "invalid symmetry scalar")
    op = build_strict_airgap(
        values["a1_node_ids"], values["a1_node_xy_mm"], a1_ids,
        values["a1_triangle_node_ids"], np.full(len(a1_ids), int(codes[0]), dtype=np.int64),
        {int(code): "a1" for code in codes}, [], [], admitted_element_ids=a1_ids,
        axial_length_m=float(values["axial_length_m"]), symmetry_multiplier=int(values["symmetry_multiplier"]),
    )
    require(np.array_equal(op.a1_orientation, values["a1_orientation"]), "a1 orientation binding differs")
    for key in ("radius_m", "area_m2", "cos_theta", "sin_theta"):
        actual, expected = getattr(op.band, key), np.asarray(values[key])
        require(actual.shape == expected.shape and np.isfinite(expected).all()
                and np.allclose(actual, expected, rtol=1e-12, atol=0), f"a1 operator mismatch: {key}")
    for key in ("r_inner_m", "r_outer_m", "sector_span_rad"):
        require(np.asarray(values[key]).shape == () and np.isfinite(values[key])
                and np.isclose(getattr(op.band, key), float(values[key]), rtol=1e-12, atol=1e-14), f"a1 scalar mismatch: {key}")
    return replace(op, admitted_element_ids=freeze(admitted),
                   band=replace(op.band, region_codes=tuple(map(int, sorted(codes)))))


def encoded_design(design_op, mechanical_angle_deg, encoding):
    d = np.asarray(design_op)
    require(d.shape == (12,) and np.isfinite(d).all(), "invalid design/OP vector")
    angle = float(mechanical_angle_deg)
    require(np.isfinite(angle), "invalid mechanical angle")
    m = np.deg2rad(angle)
    require(abs(d[10] - np.sin(m)) < 1e-5 and abs(d[11] - np.cos(m)) < 1e-5, "mechanical-angle input mismatch")
    require(encoding in ("mechanical", "electrical"), "undeclared phase encoding")
    result = d.copy()
    if encoding == "electrical":
        result[10], result[11] = np.sin(4 * m), np.cos(4 * m)
    return result


def validate_checkpoint_metadata(checkpoint, job, selection, legacy=None):
    meta = checkpoint["metadata"]
    require(checkpoint["epoch"] == 30 and meta["training_updates"] == 88565, "not the final88565 checkpoint")
    require(meta["seed"] == job["seed"] and job["seed"] in (17, 29, 43), "seed mismatch")
    require(job["id"] == f"{job['arm']}_s{job['seed']}", "run ID differs from arm/seed")
    require(job["arm"] in ARMS and job["phase_encoding"] == ARMS[job["arm"]], "arm/encoding contract mismatch")
    require(meta["old_checkpoint_reused"] is False and meta.get("status", "RUNNING") in ("RUNNING", "COMPLETED"), "checkpoint training state invalid")
    allowed = set(job.get("legacy_absent_fields", []))
    limit = {"A0_s17": {"arm", "phase_encoding"}, "A0_s29": {"phase_encoding"}}.get(job["id"], set())
    require(allowed <= limit, "unapproved metadata exception")
    missing = []
    for key in ("arm", "phase_encoding"):
        if key not in meta:
            require(key in allowed and legacy is not None, f"missing checkpoint {key}")
            require(legacy["checkpoint_sha256"] == job["checkpoint"]["sha256"]
                    and key in legacy["allowed_absent_fields"], "legacy evidence does not bind checkpoint")
            missing.append(key)
        else:
            require(meta[key] == job[key], f"checkpoint {key} mismatch")
    if job["arm"] == "A2":
        require(meta.get("pole_pairs") == 4, "A2 pole-pair metadata mismatch")
    for key, digest in job["metadata_hashes"].items():
        require(meta.get(key) == digest, f"checkpoint metadata hash mismatch: {key}")
    train = sorted({r["geometry_group"] for r in selection["records"] if r["split"] == "train"})
    dev = sorted({r["geometry_group"] for r in selection["records"] if r["split"] == "development"})
    require(len(train) == 288 and len(dev) == 32 and not set(train) & set(dev), "split group contract mismatch")
    require(meta["train_groups"] == train and sorted(meta["development_groups"]) == dev, "checkpoint group mismatch")
    if job["arm"] in ("C", "G"):
        require(meta.get("context_scope") == {"C": "component", "G": "graph"}[job["arm"]], "context scope mismatch")
    return missing


def validate_graph(arrays, operator, step):
    ids = integer_ids(arrays["native_element_id"], "graph native element IDs")
    require(np.array_equal(np.sort(ids), operator.admitted_element_ids), "graph admission ID set differs")
    native_step = np.asarray(arrays["step"])
    require(native_step.shape == () and native_step.dtype.kind in "iu" and int(native_step) == step,
            "native integer step differs from manifest filename")
    n = len(ids)
    for key, shape in (("node", (n, 14)), ("weights", (n,)), ("target_B_T", (n, 2)), ("native_reg_code", (n,))):
        require(arrays[key].shape == shape and np.isfinite(arrays[key]).all(), f"invalid graph {key}")
    require(arrays["native_reg_code"].dtype.kind in "iu", "region codes must be native integer IDs")
    require(np.all(arrays["weights"] > 0), "nonpositive graph area")
    require(arrays["native_tri_node_ids"].shape == (n, 3) and arrays["native_tri_node_ids"].dtype.kind in "iu", "invalid native incidence")
    a1_pos = exact_positions(ids, operator.a1_element_ids)
    require(np.array_equal(arrays["native_tri_node_ids"][a1_pos], operator.a1_triangle_node_ids), "a1 native incidence mismatch")
    require(np.isin(arrays["native_reg_code"][a1_pos], operator.band.region_codes).all(), "a1 native region mismatch")
    # Frozen bundle areas are float32; compare against native float64 geometry.
    require(np.allclose(arrays["weights"][a1_pos].astype(float), operator.band.area_m2,
                        rtol=4 * np.finfo(np.float32).eps, atol=0), "a1 float32 graph area differs from native geometry")
    edges = arrays["edge_index"]
    require(edges.ndim == 2 and edges.shape[0] == 2 and edges.dtype.kind in "iu", "invalid native edge IDs")
    require(np.all(edges >= 0) and np.all(edges < n), "edge outside this graph")
    require(arrays["edge_attr"].shape == (edges.shape[1], 4) and np.isfinite(arrays["edge_attr"]).all(), "invalid edge attributes")
    return a1_pos


def phase_scopes(step):
    require(isinstance(step, (int, np.integer)) and 0 <= step < 45, "invalid phase step")
    return ("all45", "seen9") if step in SEEN else ("all45", "unused36", "after_last_seen" if step > 40 else "interior_unused")


def add_field_stats(stats, key, pred, truth, weights):
    value = field_sufficient_statistics(pred, truth, weights)
    stats[key] = merge_statistics([stats[key], value]) if key in stats else value


def accumulate_snapshot(stats, operator, ids, pred, truth, weights, case, step, phase_advance):
    a1_pred = align_airgap_field(operator, ids, pred)
    a1_truth = align_airgap_field(operator, ids, truth)
    pa = "PA_ge_70" if phase_advance >= 70 else "PA_lt_70"
    for phase in phase_scopes(step):
        for group in (phase, f"{phase}/case:{case}", f"{phase}/{pa}"):
            for region, p, t, w in (("all", pred, truth, weights), ("a1", a1_pred, a1_truth, operator.band.area_m2)):
                add_field_stats(stats, f"{group}/{region}/area", p, t, w)
                add_field_stats(stats, f"{group}/{region}/unweighted", p, t, None)
    return a1_pred, a1_truth, strict_arkkio_torque(operator, ids, pred), strict_arkkio_torque(operator, ids, truth)


def pooled_torque(pred, truth):
    pred, truth = np.asarray(pred, float), np.asarray(truth, float)
    require(pred.ndim == 1 and len(pred) > 0 and pred.shape == truth.shape
            and np.isfinite(pred).all() and np.isfinite(truth).all(), "invalid pooled torque samples")
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        rmse = float(np.sqrt(np.mean((pred - truth)**2)))
        rms = float(np.sqrt(np.mean(truth**2)))
        require(np.isfinite(rmse) and np.isfinite(rms), "nonfinite pooled torque reduction")
        normalized = 100 * rmse / rms if rms > 0 else None
        require(normalized is None or np.isfinite(normalized), "nonfinite torque nRMSE")
    return {"count": len(pred), "rmse_Nm": rmse, "truth_rms_Nm": rms, "nrmse_percent": normalized,
            "normalization_status": "DEFINED" if rms > 0 else "UNDEFINED_ZERO_REFERENCE",
            "ripple": "NOT_POOLED_ACROSS_CASES"}


def reproduction_gate(stats, original_eval, original_train, tolerance=.0001):
    differences = {}
    for phase in ("all45", "seen9"):
        actual = finish_statistics(stats[f"{phase}/all/area"]).vector_nrmse_percent
        expected = original_eval["metrics"][f"{phase}/all"]["B_vector_nRMSE_percent"]
        require(actual is not None and np.isfinite(expected), "undefined reproduction anchor")
        differences[f"{phase}_original_eval_pp"] = abs(actual - expected)
    train_anchor = original_train["metrics"][-1]["development"]["all"]["vector_nRMSE_percent"]
    actual = finish_statistics(stats["seen9/all/area"]).vector_nrmse_percent
    require(np.isfinite(train_anchor), "nonfinite training anchor")
    differences["seen9_training_anchor_pp"] = abs(actual - train_anchor)
    require(all(delta < tolerance for delta in differences.values()), f"frozen score reproduction FAIL: {differences}")
    return {"status": "PASS_ARITHMETIC_ONLY", "tolerance_pp": tolerance, "absolute_differences_pp": differences}


def read_components(spec, dev_cases, metadata, check):
    path = pinned_path(spec, check)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    require(manifest["status"] == "COMPLETED" and manifest["cases"] == 360, "component manifest incomplete")
    require(spec["sha256"] == metadata["component_manifest_sha256"], "component hash differs from checkpoint")
    require(manifest["source_bundle_manifest_sha256"] == metadata["bundle_manifest_sha256"], "component source training bundle mismatch")
    result = {}
    for row in manifest["records"]:
        if row["case_index"] not in dev_cases:
            continue
        require(row["case_index"] not in result, "duplicate component case")
        f = child_file(path.parent, row["file"])
        require(f.stat().st_size == row["bytes"] and sha256(f, check) == row["sha256"], "component bytes mismatch")
        with np.load(f, allow_pickle=False) as z:
            ids, component = z["native_element_id"], z["component_id"]
        integer_ids(ids, "component native IDs")
        require(component.dtype == np.int64 and component.shape == ids.shape and len(ids) == row["triangles"], "component shape mismatch")
        require(np.array_equal(np.unique(component), np.arange(row["component_count"])), "component labels mismatch")
        result[row["case_index"]] = {"ids": ids, "component": component, "edge_sha": row["edge_index_sha256"]}
    require(set(result) == set(dev_cases), "component dev32 coverage mismatch")
    return result


def verify_sources(source_files, check):
    resolved = {Path(row["path"]).name: pinned_path(row, check) for row in source_files}
    require(Path(__file__).name in resolved and resolved[Path(__file__).name] == Path(__file__).resolve(), "evaluator is not source-pinned")
    for row in source_files:
        if "module" in row:
            module = importlib.import_module(row["module"])
            require(Path(module.__file__).resolve() == Path(row["path"]).resolve(), f"loaded module differs: {row['module']}")
    return resolved


def run_job(job, job_sha, out):
    start = time.monotonic()
    deadline = dt.datetime.fromisoformat(job["deadline_UTC"])
    require(deadline.tzinfo is not None and job["release_status"] == "RELEASED_FROZEN_EVALUATION", "job not released or deadline unzoned")
    require(job["max_seconds"] == 1800 and job["workers"] == 1 and job["retry_limit"] == 0, "resource contract changed")
    require(job["training_updates"] == job["new_FEA"] == job["new_prior_solves"] == 0, "forbidden computation in job")
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    torch = None
    result = {"status": "RUNNING", "id": job["id"], "arm": job["arm"], "seed": job["seed"],
              "job_sha256": job_sha, "checkpoint_sha256": job["checkpoint"]["sha256"],
              "graphs_processed": 0, "training_updates": 0, "new_FEA": 0, "new_prior_solves": 0,
              "automatic_retries": 0, "prediction_files": [], "case_torque": {},
              "scope": "E1-oldTRAIN-exposed innerdev32; no independent test/AC/virtual-work qualification"}
    def check():
        if time.monotonic() - start >= 1800 or dt.datetime.now(dt.timezone.utc) >= deadline:
            raise InterruptedError("frozen wall/UTC bound")
        if (out / "STOP").exists() or any(Path(p).exists() for p in job["stop_files"]):
            raise InterruptedError("STOP requested")
        if torch is not None and torch.cuda.is_initialized() and torch.cuda.max_memory_allocated() > 10 * 1024**3:
            raise InterruptedError("10 GiB torch-allocated soft limit")
    def save():
        result["elapsed_seconds"] = time.monotonic() - start
        temporary = out / "RESULT.tmp"
        temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(out / "RESULT.json")  # one atomic attempt; retry0
    save()
    try:
        check()
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        sources = verify_sources(job["source_files"], check)
        selection = pinned_json(job["selection"], check)
        bundle_root = Path(job["bundle_dir"]).resolve(strict=True)
        bundle = pinned_json({"path": str(bundle_root / "BUNDLE_MANIFEST.json"), "sha256": job["bundle_manifest_sha256"]}, check)
        require(bundle["selection_sha256"] == job["selection"]["sha256"] and bundle["reader_pass_cases"] == 32 and bundle["failed_cases"] == 0, "bundle selection/coverage mismatch")
        dev = {r["case_index"]: r for r in selection["records"] if r["split"] == "development"}
        require(len(dev) == 32 and all(r["original_split"] == "train" for r in dev.values()), "only frozen32 oldTRAIN dev admitted")
        original_eval, original_train = pinned_json(job["original_eval"], check), pinned_json(job["original_train"], check)
        require(original_eval["checkpoint_sha256"] == job["checkpoint"]["sha256"]
                and original_eval["bundle_manifest_sha256"] == job["bundle_manifest_sha256"]
                and original_eval["graphs_processed"] == 1440 and original_eval["status"] == "COMPLETED", "original evaluation binding mismatch")
        geometry_path = pinned_path(job["geometry_manifest"], check)
        geometry = json.loads(geometry_path.read_text(encoding="utf-8"))
        require(geometry["status"] == "ADMITTED" and geometry["selection_sha256"] == job["selection"]["sha256"]
                and geometry["bundle_manifest_sha256"] == job["bundle_manifest_sha256"], "geometry admission not bound")
        operators, operator_rows = {}, {}
        for row in geometry["records"]:
            case = row["case_index"]
            require(case in dev and case not in operators and row["status"] == "ADMITTED"
                    and row["source_sha256"] == dev[case]["source_sha256"] and row["steps_checked"] == 45, "geometry case admission mismatch")
            path = child_file(geometry_path.parent, row["operator_file"])
            require(sha256(path, check) == row["operator_sha256"], "operator sidecar SHA mismatch")
            operators[case] = load_operator(path); operator_rows[case] = row
            require(operators[case].axial_length_m == dev[case]["native_input"]["Stator_Lam_Length"] * .001, "native stack-length mismatch")
        require(set(operators) == set(dev), "geometry dev32 coverage incomplete")
        files = {}; pairs = []
        for record in bundle["records"]:
            case = record["case_index"]
            require(case in dev and case not in files and record["geometry_group"] == dev[case]["geometry_group"]
                    and record["source_sha256"] == dev[case]["source_sha256"] and record["original_split"] == "train"
                    and record["QA"]["all_45_steps_checked"] and len(record["files"]) == 45, "bundle case binding mismatch")
            rows = []
            for row in record["files"]:
                path = child_file(bundle_root, row["name"])
                step = int(path.stem.split("_step")[1])
                require(int(path.stem.split("_step")[0][4:]) == case, "graph filename case mismatch")
                require(path.stat().st_size == row["bytes"] and sha256(path, check) == row["sha256"], "graph bytes mismatch")
                pairs.append((case, step)); rows.append((step, path, row["sha256"]))
            files[case] = sorted(rows)
        validate_snapshot_coverage(np.array(pairs, dtype=np.int64), list(dev))
        checkpoint_path = pinned_path(job["checkpoint"], check)
        legacy = pinned_json(job["legacy_evidence"], check) if job.get("legacy_absent_fields") else None
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        import torch as torch_module
        torch = torch_module
        torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False; torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        require(torch.cuda.is_available(), "declared CUDA evaluation environment unavailable")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        result["metadata_exceptions_used"] = validate_checkpoint_metadata(checkpoint, job, selection, legacy)
        meta = checkpoint["metadata"]
        from dual_solver_mgn import DomainContract, DualSolverMGN
        group_hash = hashlib.sha256(json.dumps(meta["train_groups"]).encode()).hexdigest()
        contract = DomainContract(tuple(bundle["node_features"]), tuple(bundle["edge_features"]),
                                  ("native_element_Bx_T", "native_element_By_T"), group_hash,
                                  "native_element_reported", "native_explicit_ID")
        contracts = {domain: contract for domain in ("elmer", "motorcad")}
        components = {}
        if job["arm"] in ("C", "G"):
            from context_checkpoint import restore_context
            cg_plan = pinned_json(job["cg_plan"], check)
            model = restore_context(checkpoint, contracts, len(bundle["design_features"]), cg_plan,
                                    {"context_model_sha256": meta["context_model_sha256"], "model_sha256": meta["model_sha256"],
                                     "plan_sha256": job["cg_plan"]["sha256"], "selection_sha256": job["selection"]["sha256"]})
            components = read_components(job["components_manifest"], dev, meta, check)
            expected_parameters = 71206
        else:
            model = DualSolverMGN(contracts, len(bundle["design_features"]), ("MotorCAD_N45P1_formal_UNKNOWN",), width=32, depth=3)
            model.load_state_dict(checkpoint["model"], strict=True)
            expected_parameters = 70182
        require(sum(p.numel() for p in model.parameters()) == expected_parameters, "model parameter-count mismatch")
        require(sha256(sources["dual_solver_mgn.py"], check) == meta["model_sha256"], "loaded backbone source differs from training")
        if job["arm"] in ("C", "G"):
            require(sha256(sources["context_field_model.py"], check) == meta["context_model_sha256"], "loaded context source differs from training")
        model = model.cuda().eval(); model.requires_grad_(False)
        del checkpoint
        result["environment"] = {"python": sys.version, "torch": torch.__version__, "numpy": np.__version__,
                                 "device": torch.cuda.get_device_name(), "executable": sys.executable}
        stats = {}; torque_rows = []; observed = []
        result.update(selection_sha256=job["selection"]["sha256"], bundle_manifest_sha256=job["bundle_manifest_sha256"],
                      geometry_manifest_sha256=job["geometry_manifest"]["sha256"], phase_encoding=job["phase_encoding"])
        with torch.inference_mode():
            for case in sorted(files):
                op = operators[case]; case_pred = []; case_true = []; cp = []; ct = []; graph_hashes = []
                for step, path, graph_sha in files[case]:
                    check()
                    # Recheck immediately before model input; no stale inventory allowance.
                    require(sha256(path, check) == graph_sha, "graph changed since admission")
                    with np.load(path, allow_pickle=False) as z:
                        arrays = {key: z[key] for key in ("node", "edge_index", "edge_attr", "weights", "design_op", "target_B_T",
                                  "native_element_id", "native_tri_node_ids", "native_reg_code", "step", "mechanical_angle_deg")}
                    validate_graph(arrays, op, step)
                    graph = {key: torch.as_tensor(arrays[key], device="cuda", dtype=torch.long if key == "edge_index" else torch.float32)
                             for key in ("node", "edge_index", "edge_attr", "weights")}
                    graph["batch"] = torch.zeros(len(arrays["node"]), device="cuda", dtype=torch.long)
                    if components:
                        component = components[case]
                        require(np.array_equal(arrays["native_element_id"], component["ids"])
                                and hashlib.sha256(arrays["edge_index"].tobytes()).hexdigest() == component["edge_sha"], "context component native order/edge binding mismatch")
                        graph["component_id"] = torch.as_tensor(component["component"], device="cuda", dtype=torch.long)
                    design = encoded_design(arrays["design_op"], arrays["mechanical_angle_deg"], job["phase_encoding"])
                    pred = model.forward_fields("motorcad", graph, torch.as_tensor(design, device="cuda")[None],
                                                torch.zeros(1, device="cuda", dtype=torch.long))["fields"].cpu().numpy().astype(float)
                    require(np.isfinite(pred).all() and pred.shape == arrays["target_B_T"].shape, "invalid model field")
                    target = arrays["target_B_T"].astype(float); weights = arrays["weights"].astype(float)
                    ap, at, tp, tt = accumulate_snapshot(stats, op, arrays["native_element_id"], pred, target, weights,
                                                        case, step, dev[case]["native_input"]["PhaseAdvance"])
                    case_pred.append(ap); case_true.append(at); cp.append(tp); ct.append(tt); graph_hashes.append(graph_sha)
                    torque_rows.append({"case": case, "step": step, "pred_Nm": tp, "truth_Nm": tt})
                    observed.append((case, step)); result["graphs_processed"] += 1
                case_summary = asdict(summarize_torque45(list(range(45)), cp, ct))
                result["case_torque"][str(case)] = case_summary
                cache = out / f"a1_case{case:04d}.npz"
                np.savez_compressed(cache, case_index=np.array(case), step=np.arange(45), native_element_id=op.a1_element_ids,
                                    pred_B_T=np.stack(case_pred), truth_B_T=np.stack(case_true),
                                    checkpoint_sha256=np.array(job["checkpoint"]["sha256"]), source_sha256=np.array(dev[case]["source_sha256"]),
                                    operator_sha256=np.array(operator_rows[case]["operator_sha256"]), selection_sha256=np.array(job["selection"]["sha256"]),
                                    bundle_manifest_sha256=np.array(job["bundle_manifest_sha256"]), graph_sha256=np.array(graph_hashes))
                result["prediction_files"].append({"case_index": case, "file": cache.name, "sha256": sha256(cache, check), "bytes": cache.stat().st_size})
                save()
        validate_snapshot_coverage(np.array(observed, dtype=np.int64), list(dev))
        result["reproduction_gate"] = reproduction_gate(stats, original_eval, original_train)
        result["field_metrics"] = {key: asdict(finish_statistics(value)) for key, value in sorted(stats.items())}
        result["field_sufficient_statistics"] = {key: asdict(value) for key, value in sorted(stats.items())}
        result["torque_samples"] = torque_rows
        result["pooled_torque"] = pooled_torque([row["pred_Nm"] for row in torque_rows], [row["truth_Nm"] for row in torque_rows])
        result["mean_case_torque_nrmse_percent"] = float(np.mean([v["nrmse_percent"] for v in result["case_torque"].values()])) if all(v["nrmse_percent"] is not None for v in result["case_torque"].values()) else None
        check()
        require(sha256(checkpoint_path, check) == job["checkpoint"]["sha256"], "checkpoint changed during evaluation")
        result.update(status="COMPLETED", completed_UTC=dt.datetime.now(dt.timezone.utc).isoformat(),
                      peak_torch_allocated_bytes=torch.cuda.max_memory_allocated(), verdict="DIAGNOSTICS_ONLY_NO_NEW_IMPROVEMENT_GATE")
    except BaseException as exc:
        result.update(status="STOPPED" if isinstance(exc, InterruptedError) else "INCOMPLETE_NO_CLAIM", error=repr(exc))
        raise
    finally:
        save()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--job-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    job = pinned_json({"path": str(args.job), "sha256": args.job_sha256})
    run_job(job, args.job_sha256, args.out)


if __name__ == "__main__":
    main()
