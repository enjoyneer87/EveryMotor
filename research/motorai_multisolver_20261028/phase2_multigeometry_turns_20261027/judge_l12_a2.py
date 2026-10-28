"""Saved-results-only A2 arithmetic; preserve all three seeds and the original margins.

No model loading, data opening, training or checkpoint selection occurs here.
Missing/failed/inconsistent runs yield INCOMPLETE_NO_CLAIM, never a subset score.
"""
from pathlib import Path
import csv
import hashlib
import json
import statistics

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "master_plan_artifacts/l12a2"
SEEDS = (17, 29, 43)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(ev, rows, phase_advance, seen):
    metrics = ev["metrics"]
    cases = sorted(int(key.split(":")[1]) for key in metrics if key.startswith("all45/case:"))
    high = [case for case in cases if phase_advance[case] >= 70]
    low = [case for case in cases if phase_advance[case] < 70]
    phase_groups = {"seen9": [], "interior_unused": [], "after_last_seen": []}
    for row in rows:
        step = int(row["step"])
        group = "seen9" if step in seen else ("after_last_seen" if step > max(seen) else "interior_unused")
        phase_groups[group].append(float(row["vector_nRMSE_percent"]))
    return {
        "checkpoint_sha256": ev["checkpoint_sha256"], "seed": ev["seed"],
        "all45_vector": metrics["all45/all"]["B_vector_nRMSE_percent"],
        "all45_magnitude": metrics["all45/all"]["B_magnitude_nRMSE_percent"],
        "all45_vector_RMSE_T": metrics["all45/all"]["B_vector_RMSE_T"],
        "seen9_vector": metrics["seen9/all"]["B_vector_nRMSE_percent"],
        "unused36_vector": metrics["unused36/all"]["B_vector_nRMSE_percent"],
        "all45_copper_vector": metrics["all45/copper"]["B_vector_nRMSE_percent"],
        "copper_B2_bias": ev["copper_B2"]["all45"]["signed_bias_percent"],
        "PA_ge_70_mean_case_vector": statistics.mean(metrics[f"all45/case:{case}"]["B_vector_nRMSE_percent"] for case in high),
        "PA_lt_70_mean_case_vector": statistics.mean(metrics[f"all45/case:{case}"]["B_vector_nRMSE_percent"] for case in low),
        "phase_region_mean_snapshot_vector": {key: statistics.mean(values) for key, values in phase_groups.items()},
    }


def assess(per):
    """Same two-seed range rule extended to fixed three-seed arms; no statistics claim."""
    mean = lambda arm, key: statistics.mean(per[f"{arm}_s{seed}"][key] for seed in SEEDS)
    ranges = {arm: max(per[f"{arm}_s{seed}"]["all45_vector"] for seed in SEEDS)
                    - min(per[f"{arm}_s{seed}"]["all45_vector"] for seed in SEEDS) for arm in ("A0", "A2")}
    spread = max(ranges.values())
    delta = mean("A0", "all45_vector") - mean("A2", "all45_vector")
    harm = {key: {"change_pp": mean("A2", key) - mean("A0", key), "limit_pp": limit}
            for key, limit in (("seen9_vector", .5), ("all45_copper_vector", 1.), ("PA_ge_70_mean_case_vector", 1.))}
    for value in harm.values():
        value["pass"] = value["change_pp"] <= value["limit_pp"]
    improvement = delta >= 1.0 and delta > 2 * spread
    no_harm = all(value["pass"] for value in harm.values())
    return {"arm_means": {arm: {key: mean(arm, key) for key in (
                "all45_vector", "all45_magnitude", "seen9_vector", "unused36_vector", "all45_copper_vector", "copper_B2_bias")} for arm in ("A0", "A2")},
            "delta_all45_vector_A0_minus_A2": delta,
            "paired_deltas_by_seed": {str(seed): per[f"A0_s{seed}"]["all45_vector"] - per[f"A2_s{seed}"]["all45_vector"] for seed in SEEDS},
            "within_arm_ranges_pp": ranges, "seed_spread_all45_vector": spread,
            "range_interpretation": "descriptive observed max-min; neither best-seed selection nor a confidence interval",
            "improvement": improvement, "no_harm": harm, "no_harm_pass": no_harm,
            "verdict": "A2_IMPROVES_DEV" if improvement and no_harm else (
                "NO_IMPROVEMENT_CLAIM" if not improvement else "IMPROVES_BUT_HARM_LIMIT_EXCEEDED")}


def main():
    selection = json.loads((ROOT / "SCALE360_SELECTION_FROZEN.json").read_text(encoding="utf-8"))
    plan = json.loads((ROOT / "L12_PHASE45_PLAN.json").read_text(encoding="utf-8"))
    contract = ROOT / "master_plan_artifacts/COMMON_EVALUATION_CONTRACT.json"
    phase_advance = {row["case_index"]: row["native_input"]["PhaseAdvance"] for row in selection["records"]}
    admitted = {row["case_index"] for row in selection["records"] if row["split"] == "development"}
    per, failures = {}, {}
    for arm in ("A0", "A2"):
        for seed in SEEDS:
            run = f"{arm}_s{seed}"
            folder = ROOT / "master_plan_artifacts/l12" / ("A0_s17_ref" if seed == 17 else run) if arm == "A0" and seed != 43 else OUTPUT / "runs" / run
            ev_path = folder / "eval_RESULT.json" if arm == "A0" and seed != 43 else folder / "eval/RESULT.json"
            csv_path = folder / "snapshot_errors.csv" if arm == "A0" and seed != 43 else folder / "eval/snapshot_errors.csv"
            if not ev_path.exists() or not csv_path.exists():
                failures[run] = "missing final evaluation or snapshot CSV"
                continue
            if arm == "A2" or seed == 43:
                required = [folder / "train/RESULT.json", folder / "train_RECEIPT.json", folder / "evaluate_RECEIPT.json"]
                if not all(path.exists() for path in required):
                    failures[run] = "missing training or execution receipt"
                    continue
                train, training_receipt, evaluation_receipt = [json.loads(path.read_text(encoding="utf-8")) for path in required]
                if (train.get("status") != "COMPLETED" or train.get("training_updates") != 88565
                        or train.get("seed") != seed or train.get("old_checkpoint_reused") is not False
                        or train.get("original_val_test_used") != 0
                        or any(receipt.get("returncode") != 0 or receipt.get("stop_reason") is not None
                               for receipt in (training_receipt, evaluation_receipt))):
                    failures[run] = "training or receipt violates the fixed completion contract"
                    continue
            ev = json.loads(ev_path.read_text(encoding="utf-8"))
            rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
            if ev.get("status") != "COMPLETED" or ev.get("seed") != seed or ev.get("graphs_processed") != 1440:
                failures[run] = "evaluation not COMPLETED for the fixed seed and1440 snapshots"
                continue
            expected_encoding = "electrical" if arm == "A2" else "mechanical"
            if ev.get("phase_encoding", "mechanical") != expected_encoding:
                failures[run] = "phase encoding differs from the declared arm"
                continue
            keys = [(int(row["case"]), int(row["step"])) for row in rows]
            if set(keys) != {(case, step) for case in admitted for step in range(45)} or len(keys) != 1440:
                failures[run] = "missing/duplicate/extra case-phase rows"
                continue
            if ev["selection_sha256"] != sha(ROOT / "SCALE360_SELECTION_FROZEN.json"):
                failures[run] = "selection hash mismatch"
                continue
            if ev["model_sha256"] != sha(ROOT / "dual_solver_mgn.py"):
                failures[run] = "model code mismatch"
                continue
            per[run] = summarize(ev, rows, phase_advance, set(plan["seen_steps"]))
    result = {"experiment": "L12 A2 electrical-angle encoding, phase9, fixed3 seeds per arm",
              "evaluation_contract_sha256": sha(contract), "fixed_seeds": list(SEEDS),
              "runs": per, "missing_or_failed": failures,
              "decision": {"verdict": "INCOMPLETE_NO_CLAIM"} if failures else assess(per),
              "scope": "innerdev32, E1 oldTRAIN exposed; no independent test, AC or E1-vs-new torque comparison",
              "review_status": "Arithmetic applies original margins; final scientific interpretation belongs to Astra xHigh or higher"}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "L12_A2_RESULT.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": result["decision"]["verdict"], "complete_runs": len(per), "missing_or_failed": list(failures)}))


if __name__ == "__main__":
    main()
