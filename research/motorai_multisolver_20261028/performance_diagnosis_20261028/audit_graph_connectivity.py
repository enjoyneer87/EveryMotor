"""Read-only, train-only topology/capacity audit; no field-label access or GPU work."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def connectivity(node: np.ndarray, edge_index: np.ndarray) -> dict:
    """Find undirected components using the stored explicit-ID adjacency only."""
    parent = np.arange(len(node))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = int(parent[i])
        return i

    for left, right in edge_index.T:
        a, b = root(int(left)), root(int(right))
        if a != b:
            parent[a] = b
    labels = np.array([root(i) for i in range(len(node))])
    components = []
    names = ("stator", "rotor", "magnet", "copper", "air", "wedge", "shaft", "impreg", "other")
    for component, size in Counter(labels.tolist()).most_common():
        mask = labels == component
        materials = {name: int(np.sum(node[mask, 3 + j] > .5)) for j, name in enumerate(names)}
        components.append({"triangles": size, "materials": materials})
    stator = set(labels[node[:, 3] > .5].tolist())
    rotor = set(labels[(node[:, 4] > .5) | (node[:, 5] > .5)].tolist())
    return {
        "triangles": len(node),
        "directed_edges": edge_index.shape[1],
        "component_count": len(components),
        "isolated_triangles": sum(c["triangles"] for c in components if c["triangles"] == 1),
        "stator_component_count": len(stator),
        "rotor_or_magnet_component_count": len(rotor),
        "rotor_or_magnet_can_message_stator": bool(stator & rotor),
        "components": components,
    }


def audit(bundle: Path, code_dir: Path) -> dict:
    """Read six deterministic train records and three already-trained steps each."""
    manifest_path = bundle / "BUNDLE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    train = sorted((r for r in manifest["records"] if r["split"] == "train"), key=lambda r: r["case_index"])
    selected = [train[i] for i in np.linspace(0, len(train) - 1, 6, dtype=int)]
    rows = []
    for record in selected:
        assert record["original_split"] == "train"
        assert record["geometry_group"] not in {"R1_0199", "R1_0076"}
        for step in (0, 20, 40):
            entry = next(f for f in record["files"] if f["name"].endswith(f"_step{step:02d}.npz"))
            path = bundle / entry["name"]
            assert path.stat().st_size == entry["bytes"] and sha256(path) == entry["sha256"]
            with np.load(path, allow_pickle=False) as arrays:
                # Deliberately do not load target_B_T or any field array.
                node, edges = arrays["node"], arrays["edge_index"]
                result = connectivity(node, edges)
            rows.append({
                "case_index": record["case_index"], "geometry_group": record["geometry_group"],
                "split": "train", "step": step, "source_key": record["source_key"],
                "source_sha256": record["source_sha256"], "graph_name": entry["name"],
                "graph_sha256": entry["sha256"], **result,
            })
    # Construction on CPU only; no checkpoint deserialization, forward, gradient or CUDA call.
    import sys
    sys.path.insert(0, str(code_dir))
    from dual_solver_mgn import DomainContract, DualSolverMGN
    contract = DomainContract(tuple(manifest["node_features"]), tuple(manifest["edge_features"]),
                              ("Bx_T", "By_T"), "train_group_audit_only", "native_element_reported", "native_explicit_ID")
    model = DualSolverMGN({d: contract for d in ("elmer", "motorcad")},
                          len(manifest["design_features"]), ("MotorCAD_N45P1_formal_UNKNOWN",), width=32, depth=3)
    active_prefixes = ("encoders.motorcad.", "design.", "fidelity.", "field_condition.motorcad.")
    parameters = {name: parameter.numel() for name, parameter in model.named_parameters()}
    active = sum(count for name, count in parameters.items() if name.startswith(active_prefixes))
    return {
        "schema": "motorai.performance_diagnosis.topology.v1",
        "checked_UTC": datetime.now(timezone.utc).isoformat(),
        "host_role": "133 CPU read-only audit while existing A2 GPU jobs continue",
        "selection": "six evenly spaced sorted train condition IDs; steps0/20/40, chosen without labels",
        "bundle_manifest_sha256": sha256(manifest_path), "script_sha256": sha256(Path(__file__)),
        "model_code_sha256": sha256(code_dir / "dual_solver_mgn.py"),
        "requested_graphs": 18, "verified_graphs": len(rows),
        "graphs_without_rotor_stator_path": sum(not r["rotor_or_magnet_can_message_stator"] for r in rows),
        "capacity": {"width": 32, "depth": 3, "all_instantiated_parameters": sum(parameters.values()),
                     "parameters_on_motorcad_field_prediction_path": active,
                     "unused_by_field_loss": sum(parameters.values()) - active},
        "new_FEA": 0, "new_training_updates": 0, "GPU_inference": 0,
        "field_label_arrays_read": 0, "old_val_test_payloads_opened": 0,
        "limitations": ["18 train graphs are a topology sample, not a corpus census",
                         "No measured accuracy effect or causality from connectivity alone",
                         "Shared design/OP conditioning still reaches both components",
                         "No replacement connectivity or coordinate-nearest ID recovery performed"],
        "graphs": rows,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--code-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.bundle, args.code_dir), ensure_ascii=False))
