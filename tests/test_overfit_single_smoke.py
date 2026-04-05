"""TASK-P1A-2: Overfit-single smoke harness.

Validates that phase1_static/train.py can overfit a single synthetic sample
with --overfit-single flag and reach total_loss < overfit_target.

Design:
- All torch/physicsnemo code runs inside Docker (PhysicsNeMo container).
- Host test only orchestrates Docker exec and validates JSON output.
- Deterministic: fixed seed=42, single synthetic NPZ bundle.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest


DOCKER_CONTAINER = "physicsnemo"  # adjust if CONTAINER_NAME differs

OVERFIT_SCRIPT = """
import sys, json, tempfile, numpy as np
from pathlib import Path

try:
    import torch
    from phase1_static.motor_dataset import build_samples_from_npz

    # Build minimal synthetic NPZ with 3 nodes, 1 triangle
    angles = np.deg2rad(np.array([0.0, 22.5, 45.0], dtype=np.float64))
    inner = np.stack([np.cos(angles), np.sin(angles)], axis=1).astype(np.float32)

    pos = inner[np.newaxis, ...]                             # [1, 3, 2]
    node_type = np.ones((1, 3, 1), dtype=np.float32)
    interior_edge_index = np.array([[[0,1],[1,2],[2,0],[1,0],[2,1],[0,2]]], dtype=np.int64)
    pbc_edge_index = np.zeros((1, 2, 0), dtype=np.int64)
    pbc_edge_attr = np.zeros((1, 0, 1), dtype=np.float32)
    y = np.random.RandomState(42).randn(1, 3, 4).astype(np.float32)

    with tempfile.TemporaryDirectory() as td:
        npz_path = Path(td) / "overfit_smoke.npz"
        np.savez(
            npz_path,
            pos=pos,
            node_type_onehot=node_type,
            interior_edge_index=interior_edge_index,
            pbc_edge_index=pbc_edge_index,
            pbc_edge_attr=pbc_edge_attr,
            y=y,
        )

        import subprocess
        result = subprocess.run(
            [
                sys.executable, "-m", "phase1_static.train",
                "--input-format", "npz",
                "--data", str(npz_path),
                "--overfit-single",
                "--epochs", "150",
                "--lr", "1e-2",
                "--hidden-dim", "32",
                "--seed", "42",
                "--overfit-target", "1e-2",
            ],
            capture_output=True,
            text=True,
            cwd="/workspace",
        )
        lines = result.stdout.splitlines()
        last_loss_line = [l for l in lines if "epoch=" in l and "train" in l]
        out = {
            "returncode": result.returncode,
            "last_train_log": last_loss_line[-1] if last_loss_line else "",
            "stderr_tail": result.stderr[-300:] if result.stderr else "",
        }
        print(json.dumps(out))

except Exception as exc:
    print(json.dumps({"error": str(exc)}))
"""


def _docker_available(container: str) -> bool:
    """Check if the Docker container is running."""
    try:
        r = subprocess.run(
            ["docker", "inspect", "--format", "{{.State.Running}}", container],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return r.returncode == 0 and "true" in r.stdout.lower()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


@pytest.mark.skipif(
    not _docker_available(DOCKER_CONTAINER),
    reason=f"Docker container '{DOCKER_CONTAINER}' not running",
)
def test_overfit_single_converges_in_docker():
    """Run overfit-single inside Docker and confirm returncode 0."""
    cp = subprocess.run(
        ["docker", "exec", DOCKER_CONTAINER, "python", "-c", OVERFIT_SCRIPT],
        capture_output=True,
        text=True,
        timeout=300,
    )

    assert cp.returncode == 0, f"Docker exec failed:\n{cp.stderr[:800]}"

    stdout = cp.stdout.strip()
    # Find JSON line
    for line in stdout.splitlines():
        if line.startswith("{"):
            result = json.loads(line)
            if "error" in result:
                pytest.fail(f"Script error inside Docker: {result['error']}")
            assert result["returncode"] == 0, (
                f"train.py --overfit-single failed inside Docker:\n"
                f"{result.get('stderr_tail', '')}"
            )
            return

    pytest.fail(f"No JSON output from Docker exec:\n{stdout[:400]}")
