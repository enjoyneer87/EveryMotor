"""Hash-bound zero-update evaluator for the six new L0/L05 final checkpoints.

The common a1/Arkkio arithmetic is reused unchanged. New outputs add the original
copper B2 reduction and distinguish PA pooled values from mean-case values.
No historical A0 score is an acceptance threshold for a newly trained model.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import evaluate_common_airgap as common
from strict_airgap import finish_statistics, field_sufficient_statistics, summarize_torque45, validate_snapshot_coverage

ARMS = {"L0": 0.0, "L05": 0.5}
PHASE_COUNTS = {"all45": 45, "seen9": 9, "unused36": 36, "interior_unused": 32, "after_last_seen": 4}
HASH_KEYS = ("model_sha256", "trainer_sha256", "loss_module_sha256", "plan_sha256", "selection_sha256",
             "bundle_manifest_sha256", "mask_manifest_sha256", "initial_tensor_state_sha256", "data_order_LR_trace_sha256")
require = common.require


def valid_hash(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def validate_new_metadata(checkpoint, job, selection, train):
    meta = checkpoint["metadata"]
    arm, seed = job["arm"], job["seed"]
    require(arm in ARMS and type(seed) is int and seed in (17, 29, 43), "unregistered arm/seed")
    require(job["id"] == f"{arm}_s{seed}" and job["alpha"] == ARMS[arm]
            and job["phase_encoding"] == "mechanical", "job identity/alpha/encoding differs")
    require(type(job["alpha"]) in (int, float) and not isinstance(job["alpha"], bool), "invalid alpha type")
    require(checkpoint["epoch"] == 30 and checkpoint.get("epoch_equivalent", 30) == 30, "not final epoch30")
    for obj, label in ((meta, "checkpoint"), (train, "training result")):
        require(obj["status"] == "COMPLETED" and obj["training_updates"] == 88565, f"{label} is not final88565")
        require(obj["arm"] == arm and obj["seed"] == seed and obj["alpha"] == ARMS[arm]
                and type(obj["alpha"]) in (int, float) and not isinstance(obj["alpha"], bool), f"{label} arm/alpha/seed differs")
        require(obj["data_arm"] == "A0" and obj["phase_encoding"] == "mechanical"
                and obj["old_checkpoint_reused"] is False, f"{label} altered/reused baseline")
        require(obj["train_steps"] == sorted(common.SEEN) and obj["train_graphs"] == 2952
                and obj["train_conditions"] == 328 and obj["development_conditions"] == 32, f"{label} phase/count differs")
        require(obj.get("new_FEA") == 0 and obj.get("original_val_test_used") == 0, f"{label} source scope differs")
        for key in HASH_KEYS:
            require(key in job["metadata_hashes"] and valid_hash(job["metadata_hashes"][key])
                    and obj.get(key) == job["metadata_hashes"][key], f"{label} metadata SHA differs: {key}")
    require(train["checkpoint_sha256"] == job["checkpoint"]["sha256"], "training-result checkpoint binding differs")
    require(job["metadata_hashes"]["selection_sha256"] == job["selection"]["sha256"]
            and job["metadata_hashes"]["mask_manifest_sha256"] == job["training_mask_manifest"]["sha256"], "job selection/mask binding differs")
    groups = {s: sorted({r["geometry_group"] for r in selection["records"] if r["split"] == s}) for s in ("train", "development")}
    require(len(groups["train"]) == 288 and len(groups["development"]) == 32
            and not set(groups["train"]) & set(groups["development"]), "frozen group split differs")
    for obj in (meta, train):
        require(obj["train_groups"] == groups["train"] and sorted(obj["development_groups"]) == groups["development"], "metadata split differs")
    last = train["metrics"][-1]
    require(last["epoch"] == 30 and last["training_updates"] == 88565
            and last["development"]["all"]["snapshots"] == 288, "final seen9 anchor coverage differs")
    return meta


def validate_mask_manifest(mask, job, selection):
    require(mask["status"] == "ADMITTED" and mask["selection_sha256"] == job["selection"]["sha256"]
            and mask["bundle_manifest_sha256"] == job["metadata_hashes"]["bundle_manifest_sha256"]
            and mask["dev_geometry_manifest_sha256"] == job["geometry_manifest"]["sha256"]
            and mask["train_steps"] == sorted(common.SEEN), "training mask admission binding differs")
    for split, key, count in (("train", "records", 328), ("development", "development_records", 32)):
        expected = {r["case_index"]: r for r in selection["records"] if r["split"] == split}
        rows = mask[key]
        require(len(rows) == len(expected) == count and {r["case_index"] for r in rows} == set(expected), "mask case coverage differs")
        for row in rows:
            source = expected[row["case_index"]]
            require(row["split"] == split and row["source_sha256"] == source["source_sha256"]
                    and row["geometry_group"] == source["geometry_group"], "mask source/split differs")
            require(len(row["graphs"]) == 9 and {g["step"] for g in row["graphs"]} == common.SEEN, "mask phase coverage differs")


def accumulate_copper(stats, conductors, pred, truth, weights, node, reg, case, step):
    require(node.shape == (len(weights), 14) and np.isfinite(node).all(), "invalid material features")
    require(np.isin(node[:, 6], (0., 1.)).all(), "copper material channel is not binary")
    copper = node[:, 6] > .5
    require(copper.any() and reg.shape == (len(weights),) and reg.dtype.kind in "iu", "missing/invalid native copper")
    for phase in common.phase_scopes(step):
        for group in (phase, f"{phase}/case:{case}"):
            common.add_field_stats(stats, f"{group}/copper/area", pred[copper], truth[copper], weights[copper])
            common.add_field_stats(stats, f"{group}/copper/unweighted", pred[copper], truth[copper], None)
    for code in np.unique(reg[copper]):
        selected = copper & (reg == code)
        area = float(weights[selected].sum())
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            pb2 = float(np.sum(weights[selected, None] * pred[selected] ** 2) / area)
            tb2 = float(np.sum(weights[selected, None] * truth[selected] ** 2) / area)
            pm2 = float(np.sum((np.sum(weights[selected, None] * pred[selected], axis=0) / area) ** 2))
            tm2 = float(np.sum((np.sum(weights[selected, None] * truth[selected], axis=0) / area) ** 2))
        require(np.isfinite([area, pb2, tb2, pm2, tm2]).all() and area > 0, "nonfinite copper B2")
        require(pm2 <= pb2 + 1e-9 and tm2 <= tb2 + 1e-9, "copper Jensen consistency failed")
        for phase in common.phase_scopes(step):
            key = (int(case), int(code), phase)
            conductors[key] = conductors.get(key, np.zeros(6)) + np.array([pb2, tb2, pm2, tm2, area, 1.])


def finish_copper(conductors, expected_cases):
    require(conductors, "no copper statistics")
    require({key[0] for key in conductors} == set(expected_cases), "copper case coverage differs")
    rows, totals = [], {}
    case_codes = {(case, code) for case, code, phase in conductors}
    for case, code in sorted(case_codes):
        require({phase for c, k, phase in conductors if c == case and k == code} == set(PHASE_COUNTS), "copper phase scopes differ")
    for (case, code, phase), values in sorted(conductors.items()):
        pb2, tb2, pm2, tm2, area, n = values
        require(n == PHASE_COUNTS[phase], "copper phase count differs")
        pb2, tb2, pm2, tm2, area = (float(x / n) for x in (pb2, tb2, pm2, tm2, area))
        rows.append(dict(case=case, native_reg_code=code, phase_scope=phase, snapshots=int(n), mean_area_m2=area,
                         pred_spatial_mean_B2_T2=pb2, true_spatial_mean_B2_T2=tb2,
                         pred_square_spatial_mean_B_T2=pm2, true_square_spatial_mean_B_T2=tm2,
                         B2_signed_relative_percent=100 * (pb2 - tb2) / tb2 if tb2 else None))
        totals[phase] = totals.get(phase, np.zeros(3)) + np.array([area * abs(pb2 - tb2), area * (pb2 - tb2), area * tb2])
    summaries = {}
    for phase, (absolute, signed, reference) in totals.items():
        require(np.isfinite([absolute, signed, reference]).all() and reference >= 0, "nonfinite copper reduction")
        summaries[phase] = dict(area_weighted_relative_MAE_percent=100 * absolute / reference if reference else None,
                                signed_bias_percent=100 * signed / reference if reference else None,
                                weighted_absolute_error_T2_m2=float(absolute), weighted_signed_error_T2_m2=float(signed),
                                weighted_reference_T2_m2=float(reference),
                                normalization_status="DEFINED" if reference else "UNDEFINED_ZERO_REFERENCE",
                                reduction="uniform snapshots per native conductor, then mean-area weighting; not AC loss")
    return rows, summaries


def mean_case_metrics(stats, dev):
    result = {}
    for phase in PHASE_COUNTS:
        for label, cases in (("PA_ge_70", [c for c, r in dev.items() if r["native_input"]["PhaseAdvance"] >= 70]),
                             ("PA_lt_70", [c for c, r in dev.items() if r["native_input"]["PhaseAdvance"] < 70])):
            require(cases, "registered PA group is empty")
            for region in ("all", "a1", "copper"):
                values = [finish_statistics(stats[f"{phase}/case:{case}/{region}/area"]).vector_nrmse_percent for case in sorted(cases)]
                result[f"{phase}/{label}/{region}"] = dict(case_ids=sorted(cases), case_count=len(cases),
                    mean_case_Bvector_nrmse_percent=float(np.mean(values)) if all(v is not None for v in values) else None,
                    normalization_status="DEFINED" if all(v is not None for v in values) else "UNDEFINED_NO_SILENT_DROP",
                    reduction="arithmetic mean of per-case pooled phase B-vector nRMSE; not pooled PA nRMSE")
    return result


def new_reproduction_gate(stats, train, tolerance=.0001):
    actual = finish_statistics(stats["seen9/all/area"]).vector_nrmse_percent
    anchor = train["metrics"][-1]["development"]["all"]["vector_nRMSE_percent"]
    require(actual is not None and np.isfinite(anchor), "undefined final-training anchor")
    difference = abs(actual - anchor)
    require(difference < tolerance, f"new final checkpoint seen9 reproduction failed: {difference}")
    return {"status": "PASS_ARITHMETIC_ONLY", "absolute_differences_pp": {"seen9_training_anchor_pp": difference},
            "tolerance_pp": tolerance, "all45_historical_reproduction": "NOT_REQUIRED_FOR_NEW_MODEL"}


def historical_reference_differences(stats, pooled, reference, job):
    require(job["arm"] == "L0" and reference["status"] == "COMPLETED" and reference["arm"] == "A0"
            and reference["seed"] == job["seed"] and reference["graphs_processed"] == 1440, "historical reference identity differs")
    require(reference["selection_sha256"] == job["selection"]["sha256"]
            and reference["bundle_manifest_sha256"] == job["bundle_manifest_sha256"]
            and reference["geometry_manifest_sha256"] == job["geometry_manifest"]["sha256"], "historical reference data/operator differs")
    diff = {}
    for region in ("all", "a1"):
        key = f"all45/{region}/area"
        current = finish_statistics(stats[key])
        for metric in ("vector_nrmse_percent", "magnitude_nrmse_percent"):
            old, new = reference["field_metrics"][key][metric], getattr(current, metric)
            diff[f"{key}/{metric}"] = None if old is None or new is None else new - old
    old, new = reference["pooled_torque"]["nrmse_percent"], pooled["nrmse_percent"]
    diff["torque_pooled_nrmse_percent"] = None if old is None or new is None else new - old
    return {"status": "REPORT_ONLY_NO_GATE", "direction": "new L0 minus historical A0", "differences_pp": diff,
            "historical_checkpoint_sha256": reference["checkpoint_sha256"]}


def run_job(job, job_sha, out):
    start = time.monotonic()
    deadline = dt.datetime.fromisoformat(job["deadline_UTC"])
    require(deadline.tzinfo is not None and job["release_status"] == "RELEASED_FROZEN_EVALUATION", "unreleased/unzoned job")
    require(job["max_seconds"] == 1800 and job["workers"] == 1 and job["retry_limit"] == 0, "evaluation resource contract differs")
    require(job["training_updates"] == job["new_FEA"] == job["new_prior_solves"] == 0, "forbidden computation")
    require(job["arm"] in ARMS and job["alpha"] == ARMS[job["arm"]], "unregistered loss allocation")
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    torch = None
    result = dict(status="RUNNING", id=job["id"], arm=job["arm"], alpha=job["alpha"], seed=job["seed"], job_sha256=job_sha,
                  checkpoint_sha256=job["checkpoint"]["sha256"], graphs_processed=0, training_updates=0, new_FEA=0,
                  new_prior_solves=0, automatic_retries=0, prediction_files=[], case_torque={}, snapshot_errors=[],
                  scope="E1-oldTRAIN-exposed innerdev32; six-run preregistered loss-allocation comparison, not independent/AC/physical-torque qualification")
    def check():
        if time.monotonic() - start >= 1800 or dt.datetime.now(dt.timezone.utc) >= deadline:
            raise InterruptedError("evaluation wall/UTC bound")
        if (out / "STOP").exists() or any(Path(p).exists() for p in job["stop_files"]):
            raise InterruptedError("STOP")
        if torch is not None and torch.cuda.is_initialized() and torch.cuda.max_memory_allocated() > 10 * 1024**3:
            raise InterruptedError("10 GiB allocated-memory soft limit")
    def save():
        result["elapsed_seconds"] = time.monotonic() - start
        temporary = out / "RESULT.tmp"
        temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(out / "RESULT.json")
    save()
    try:
        check(); os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        sources = common.verify_sources(job["source_files"], check)
        require(sources.get(Path(__file__).name) == Path(__file__).resolve(), "new evaluator is not source-pinned")
        selection = common.pinned_json(job["selection"], check)
        train = common.pinned_json(job["original_train"], check)
        masks = common.pinned_json(job["training_mask_manifest"], check)
        validate_mask_manifest(masks, job, selection)
        bundle_root = Path(job["bundle_dir"]).resolve(strict=True)
        bundle = common.pinned_json({"path": str(bundle_root / "BUNDLE_MANIFEST.json"), "sha256": job["bundle_manifest_sha256"]}, check)
        require(bundle["selection_sha256"] == job["selection"]["sha256"] and bundle["reader_pass_cases"] == 32
                and bundle["failed_cases"] == 0 and len(bundle["records"]) == 32, "dev bundle admission differs")
        dev = {r["case_index"]: r for r in selection["records"] if r["split"] == "development"}
        require(len(dev) == 32 and all(r["original_split"] == "train" for r in dev.values()), "only frozen innerdev32 admitted")
        geometry_path = common.pinned_path(job["geometry_manifest"], check)
        geometry = json.loads(geometry_path.read_text(encoding="utf-8"))
        require(geometry["status"] == "ADMITTED" and geometry["selection_sha256"] == job["selection"]["sha256"]
                and geometry["bundle_manifest_sha256"] == job["bundle_manifest_sha256"], "dev geometry admission differs")
        operators, operator_rows = {}, {}
        for row in geometry["records"]:
            case = row["case_index"]
            require(case in dev and case not in operators and row["status"] == "ADMITTED"
                    and row["source_sha256"] == dev[case]["source_sha256"] and row["steps_checked"] == 45, "geometry case binding differs")
            path = common.child_file(geometry_path.parent, row["operator_file"])
            require(common.sha256(path, check) == row["operator_sha256"], "operator bytes differ")
            operators[case] = common.load_operator(path); operator_rows[case] = row
            require(operators[case].axial_length_m == dev[case]["native_input"]["Stator_Lam_Length"] * .001, "native stack length differs")
        require(set(operators) == set(dev), "incomplete geometry coverage")
        files, pairs = {}, []
        for record in bundle["records"]:
            case = record["case_index"]
            require(case in dev and case not in files and record["geometry_group"] == dev[case]["geometry_group"]
                    and record["source_sha256"] == dev[case]["source_sha256"] and record["original_split"] == "train"
                    and record["QA"]["all_45_steps_checked"] and len(record["files"]) == 45, "graph source binding differs")
            rows = []
            for row in record["files"]:
                path = common.child_file(bundle_root, row["name"]); step = int(path.stem.split("_step")[1])
                require(int(path.stem.split("_step")[0][4:]) == case, "graph filename case differs")
                require(path.stat().st_size == row["bytes"] and common.sha256(path, check) == row["sha256"], "graph bytes differ")
                pairs.append((case, step)); rows.append((step, path, row["sha256"]))
            files[case] = sorted(rows)
        validate_snapshot_coverage(np.asarray(pairs, np.int64), list(dev))
        checkpoint_path = common.pinned_path(job["checkpoint"], check)
        import torch as torch_module
        torch = torch_module
        torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False; torch.use_deterministic_algorithms(True)
        require(torch.cuda.is_available(), "CUDA unavailable")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        meta = validate_new_metadata(checkpoint, job, selection, train)
        for filename, key in (("dual_solver_mgn.py", "model_sha256"), ("train_a1_allocation.py", "trainer_sha256"),
                              ("a1_allocation_loss.py", "loss_module_sha256")):
            require(filename in sources and common.sha256(sources[filename], check) == meta[key], "training/model source bytes differ")
        from dual_solver_mgn import DomainContract, DualSolverMGN
        group_hash = hashlib.sha256(json.dumps(meta["train_groups"]).encode()).hexdigest()
        contract = DomainContract(tuple(bundle["node_features"]), tuple(bundle["edge_features"]),
                                  ("native_element_Bx_T", "native_element_By_T"), group_hash, "native_element_reported", "native_explicit_ID")
        model = DualSolverMGN({d: contract for d in ("elmer", "motorcad")}, len(bundle["design_features"]),
                              ("MotorCAD_N45P1_formal_UNKNOWN",), width=32, depth=3)
        model.load_state_dict(checkpoint["model"], strict=True)
        require(sum(p.numel() for p in model.parameters()) == 70182, "baseline parameter count differs")
        model = model.cuda().eval(); model.requires_grad_(False)
        result.update(metadata_hashes={k: meta[k] for k in HASH_KEYS}, selection_sha256=job["selection"]["sha256"],
                      evaluator_sha256=common.sha256(__file__, check), model_sha256=meta["model_sha256"],
                      bundle_manifest_sha256=job["bundle_manifest_sha256"], geometry_manifest_sha256=job["geometry_manifest"]["sha256"],
                      training_mask_manifest_sha256=job["training_mask_manifest"]["sha256"], original_train_sha256=job["original_train"]["sha256"],
                      phase_encoding="mechanical", environment={"torch": torch.__version__, "numpy": np.__version__, "device": torch.cuda.get_device_name()})
        del checkpoint
        stats, conductors, torques, observed = {}, {}, [], []
        with torch.inference_mode():
            for case in sorted(files):
                op = operators[case]; case_pred, case_true, cp, ct, graph_hashes = [], [], [], [], []
                for step, path, graph_sha in files[case]:
                    check(); require(common.sha256(path, check) == graph_sha, "graph changed after admission")
                    with np.load(path, allow_pickle=False) as z:
                        arrays = {k: z[k] for k in ("node", "edge_index", "edge_attr", "weights", "design_op", "target_B_T",
                                                   "native_element_id", "native_tri_node_ids", "native_reg_code", "step", "mechanical_angle_deg")}
                    common.validate_graph(arrays, op, step)
                    graph = {k: torch.as_tensor(arrays[k], device="cuda", dtype=torch.long if k == "edge_index" else torch.float32)
                             for k in ("node", "edge_index", "edge_attr", "weights")}
                    graph["batch"] = torch.zeros(len(arrays["node"]), device="cuda", dtype=torch.long)
                    design = common.encoded_design(arrays["design_op"], arrays["mechanical_angle_deg"], "mechanical")
                    pred = model.forward_fields("motorcad", graph, torch.as_tensor(design, device="cuda")[None],
                                                torch.zeros(1, device="cuda", dtype=torch.long))["fields"].cpu().numpy().astype(float)
                    target, weights = arrays["target_B_T"].astype(float), arrays["weights"].astype(float)
                    require(np.isfinite(pred).all() and pred.shape == target.shape, "invalid predicted field")
                    ap, at, tp, tt = common.accumulate_snapshot(stats, op, arrays["native_element_id"], pred, target, weights,
                                                               case, step, dev[case]["native_input"]["PhaseAdvance"])
                    accumulate_copper(stats, conductors, pred, target, weights, arrays["node"], arrays["native_reg_code"], case, step)
                    single = finish_statistics(field_sufficient_statistics(pred, target, weights))
                    result["snapshot_errors"].append(dict(case=case, step=step, phase_scope="seen9" if step in common.SEEN else "unused36",
                                                         vector_nrmse_percent=single.vector_nrmse_percent, magnitude_nrmse_percent=single.magnitude_nrmse_percent))
                    case_pred.append(ap); case_true.append(at); cp.append(tp); ct.append(tt); graph_hashes.append(graph_sha)
                    torques.append(dict(case=case, step=step, pred_Nm=tp, truth_Nm=tt)); observed.append((case, step))
                    result["graphs_processed"] += 1
                result["case_torque"][str(case)] = asdict(summarize_torque45(list(range(45)), cp, ct))
                cache = out / f"a1_case{case:04d}.npz"
                np.savez_compressed(cache, case_index=np.array(case), step=np.arange(45), native_element_id=op.a1_element_ids,
                    pred_B_T=np.stack(case_pred), truth_B_T=np.stack(case_true), checkpoint_sha256=np.array(job["checkpoint"]["sha256"]),
                    source_sha256=np.array(dev[case]["source_sha256"]), operator_sha256=np.array(operator_rows[case]["operator_sha256"]),
                    selection_sha256=np.array(job["selection"]["sha256"]), bundle_manifest_sha256=np.array(job["bundle_manifest_sha256"]),
                    graph_sha256=np.array(graph_hashes))
                result["prediction_files"].append(dict(case_index=case, file=cache.name, sha256=common.sha256(cache, check), bytes=cache.stat().st_size))
                save()
        validate_snapshot_coverage(np.asarray(observed, np.int64), list(dev))
        result["reproduction_gate"] = new_reproduction_gate(stats, train)
        result["field_metrics"] = {k: asdict(finish_statistics(v)) for k, v in sorted(stats.items())}
        result["field_sufficient_statistics"] = {k: asdict(v) for k, v in sorted(stats.items())}
        result["conductor_B2"], result["copper_B2"] = finish_copper(conductors, dev)
        result["PA_mean_case_metrics"] = mean_case_metrics(stats, dev)
        result["torque_samples"] = torques
        result["pooled_torque"] = common.pooled_torque([r["pred_Nm"] for r in torques], [r["truth_Nm"] for r in torques])
        values = [v["nrmse_percent"] for v in result["case_torque"].values()]
        result["mean_case_torque_nrmse_percent"] = float(np.mean(values)) if all(v is not None for v in values) else None
        result["historical_A0_reference"] = historical_reference_differences(stats, result["pooled_torque"],
            common.pinned_json(job["historical_reference"], check), job) if job.get("historical_reference") else {"status": "NOT_REQUESTED"}
        check(); require(common.sha256(checkpoint_path, check) == job["checkpoint"]["sha256"], "checkpoint changed during evaluation")
        result.update(status="COMPLETED", completed_UTC=dt.datetime.now(dt.timezone.utc).isoformat(),
                      peak_torch_allocated_bytes=torch.cuda.max_memory_allocated(), verdict="PENDING_ALL_SIX_PREREGISTERED_JUDGMENT")
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
    job = common.pinned_json({"path": str(args.job), "sha256": args.job_sha256})
    run_job(job, args.job_sha256, args.out)


if __name__ == "__main__":
    main()
