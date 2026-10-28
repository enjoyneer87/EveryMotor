"""Summarize saved research evidence without solver, training or model inference."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PHASE = ROOT.parent / "phase2_multigeometry_turns_20261027"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build() -> dict:
    topology = read_json(ROOT / "GRAPH_TOPOLOGY_AUDIT.json")
    l12 = read_json(PHASE / "master_plan_artifacts/l12/L12_PHASE45_RESULT.json")
    frozen = read_json(PHASE / "FROZEN_FIELD_DIAGNOSTIC_RESULT.json")
    rows = []
    for name, row in l12["runs"].items():
        rows.append({"run": name, **{key: row[key] for key in (
            "all45_vector", "all45_magnitude", "seen9_vector", "unused36_vector",
            "all45_copper_vector", "copper_B2_bias", "checkpoint_sha256")}})
    files = [
        ROOT / "audit_graph_connectivity.py", ROOT / "GRAPH_TOPOLOGY_AUDIT.json",
        PHASE / "dual_solver_mgn.py", PHASE / "native_element_graph.py",
        PHASE / "train_field_scale360.py", PHASE / "train_field_l12.py",
        PHASE / "evaluate_field_l12.py", PHASE / "SCALE360_SELECTION_FROZEN.json",
        PHASE / "PRIOR_MOTORAI_COMPARISON_KO.md", PHASE / "FROZEN_FIELD_DIAGNOSTIC_RESULT.json",
        PHASE / "MASTER_DEVELOPMENT_PLAN.json",
        PHASE / "master_plan_artifacts/COMMON_EVALUATION_CONTRACT.json",
        PHASE / "master_plan_artifacts/l12/L12_PHASE45_RESULT.json",
        PHASE / "master_plan_artifacts/L12_PHASE45_REPORT_KO.md",
        PHASE / "master_plan_artifacts/A2_ELECTRICAL_PERIOD_AUDIT.json",
        PHASE / "master_plan_artifacts/A2_ELECTRICAL_ANGLE_RUN_CONTRACT.json",
        PHASE / "L12_A2_PLAN.json", PHASE / "guard_l12_a2.py",
        ROOT.parents[2] / "results/logs/run_e1_airgap_docker.sh",
        ROOT.parents[2] / "train_doe_curl_mgn.py", ROOT.parents[2] / "eval/metrics.py",
    ]
    sources = [{"source_key": path.relative_to(ROOT.parents[2]).as_posix(),
                "sha256": digest(path), "bytes": path.stat().st_size} for path in files]
    return {
        "schema": "motorai.performance_diagnosis.evidence.v1",
        "generated_UTC": datetime.now(timezone.utc).isoformat(),
        "purpose": "Compare metric/model contracts and audit an implementation limitation; not a common-test scorecard",
        "old_E1": {
            "legacy6": {"B_magnitude_nRMSE_percent": 11.096697,
                        "airgap_B_magnitude_nRMSE_percent": 4.956319,
                        "torque_nRMSE_percent": 3.214573},
            "historical_scorecard_sha256": "74cfb960df989da640e6104d501ac19dc9f549a434c7e7b610296962e87ccfa8",
            "score_provenance_commit": "61ffd1bf040176116ec18536bd1805221514fc3f",
            "best_checkpoint_sha256": "f58636620d5601f202d0203bff06d3f29c601e2885811c271edded28123bcba7",
            "best_epoch": 47, "epochs_trained": 50, "train_conditions": 450,
            "width": 256, "processor_depth": 15,
            "target": "nodal Bx/By averaged to native elements (not nodal A curl)",
            "recipe": "linear solve prior, airgap weight5, band spectral weight30, all45 snapshots",
            "historical_inclusion": "Saved aggregate evidence only; no old val/test payload or weights opened in this audit",
        },
        "recent_field_models": rows,
        "frozen_reference": {"all45_vector": frozen["metrics"]["all45/all"]["B_vector_nRMSE_percent"],
                             "all45_magnitude": frozen["metrics"]["all45/all"]["B_magnitude_nRMSE_percent"]},
        "current_airgap_only_nRMSE_percent": None,
        "current_torque_nRMSE_percent": None,
        "common_E1_vs_recent_regression_verdict": "UNKNOWN_not_measured_on_same_cases_support_weights_and_quantities",
        "L12_phase45_verdict": l12["decision"]["verdict"],
        "topology": {key: topology[key] for key in (
            "verified_graphs", "graphs_without_rotor_stator_path", "capacity", "execution")},
        "interpretation_limits": [
            "Disconnected native graph paths are confirmed in a train-only sample; accuracy attribution requires an ablation",
            "Both components remain conditioned by a shared design/OP vector",
            "Parameter count is a construction difference, not an error attribution or an expressivity measurement",
            "Existing innerdev32 was exposed to E1 training and is not a new independent geometry test",
            "No E1 weights, teacher, distillation, new prior solve or cross-solver paired alignment was admitted",
            "No new FEA, training, inference or old val/test payload access was performed by this diagnostic",
        ],
        "source_manifest": sources,
    }


if __name__ == "__main__":
    result = build()
    (ROOT / "DIAGNOSTIC_EVIDENCE.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"source_files": len(result["source_manifest"]), "L12_phase45_verdict": result["L12_phase45_verdict"],
                      "current_torque": result["current_torque_nRMSE_percent"], "topology_graphs": result["topology"]["verified_graphs"]}))
