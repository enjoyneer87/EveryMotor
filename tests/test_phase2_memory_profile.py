"""Phase 2 memory leak profiler — torch-free path check on host.

Full GPU memory test must run inside Docker (physicsnemo container).
This file covers:
  1. Host-side: 100-batch numpy collation loop — no memory growth expected
  2. Docker-gated: GPU memory increase < 200 MB over 100 batches (skipped if container absent)

Run on host:
    pytest tests/test_phase2_memory_profile.py -v

Run in Docker (full GPU check):
    docker exec physicsnemo python -m pytest /workspace/app/tests/test_phase2_memory_profile.py -v -k gpu
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from phase2_dynamic.temporal_sample import TemporalMotorSample
from phase2_dynamic.collate import collate_temporal_batch


# ─── helpers ────────────────────────────────────────────────────────────────

def _make_sample(T: int = 4, N: int = 50, seed: int = 0) -> TemporalMotorSample:
    rng = np.random.default_rng(seed)
    E_s, E_d, F_node, F_edge = 200, 40, 6, 1
    return TemporalMotorSample(
        pos=rng.random((N, 2)).astype(np.float32),
        static_edge_index=rng.integers(0, N, size=(2, E_s)).astype(np.int64),
        dynamic_edge_index=rng.integers(0, N, size=(T, 2, E_d)).astype(np.int64),
        edge_attr=rng.random((T, E_s + E_d, F_edge)).astype(np.float32),
        x=rng.random((T, N, F_node)).astype(np.float32),
        y=rng.random((T, N, 4)).astype(np.float32),
        rotor_angles_deg=np.linspace(0.0, 45.0, T),
        case_id=f"mem_test_{seed:04d}",
        sequence_len=T,
    )


def _docker_available(container: str = "physicsnemo") -> bool:
    try:
        result = subprocess.run(
            ["docker", "inspect", "--format", "{{.State.Running}}", container],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() == "true"
    except Exception:
        return False


# ─── host-side numpy collation memory test ──────────────────────────────────

def test_collation_loop_no_accumulation():
    """100-batch collation loop must not accumulate unreleased Python objects."""
    import tracemalloc

    tracemalloc.start()
    snapshot_before = tracemalloc.take_snapshot()

    for i in range(100):
        samples = [_make_sample(seed=i * 4 + j) for j in range(4)]
        batch = collate_temporal_batch(samples)
        # Explicitly release — simulate DataLoader worker cycle
        del batch
        del samples

    snapshot_after = tracemalloc.take_snapshot()
    tracemalloc.stop()

    top_stats = snapshot_after.compare_to(snapshot_before, "lineno")
    # Sum net increase in bytes across all lines
    net_increase_mb = sum(s.size_diff for s in top_stats) / (1024 * 1024)

    # Allow generous 50 MB for Python internals/bytecode caching
    assert net_increase_mb < 50.0, (
        f"Memory increased by {net_increase_mb:.1f} MB over 100 collation batches. "
        "Possible accumulation leak."
    )


def test_samples_are_not_cached_between_iterations():
    """Each _make_sample call produces an independent object (no shared mutable state)."""
    s1 = _make_sample(seed=0)
    s2 = _make_sample(seed=0)
    # Same seed → same values, but different array objects (no aliasing)
    assert s1.pos is not s2.pos
    np.testing.assert_array_equal(s1.pos, s2.pos)  # same values
    assert s1.y is not s2.y


# ─── Docker-gated GPU memory test ───────────────────────────────────────────

@pytest.mark.skipif(not _docker_available(), reason="physicsnemo container not running")
def test_gpu_memory_increase_under_100_batches():
    """GPU memory must not increase > 200 MB over 100 sequential DataLoader-style batches.

    Runs inside Docker via subprocess; parses JSON result line from stdout.
    """
    import json

    script = """
import json, sys
sys.path.insert(0, '/workspace/app')

import torch
import numpy as np

from phase2_dynamic.temporal_sample import TemporalMotorSample
from phase2_dynamic.collate import collate_temporal_batch

def _make(T=4, N=50, seed=0):
    rng = np.random.default_rng(seed)
    E_s, E_d = 200, 40
    return TemporalMotorSample(
        pos=rng.random((N,2)).astype('f4'),
        static_edge_index=rng.integers(0,N,(2,E_s)).astype('i8'),
        dynamic_edge_index=rng.integers(0,N,(T,2,E_d)).astype('i8'),
        edge_attr=rng.random((T,E_s+E_d,1)).astype('f4'),
        x=rng.random((T,N,6)).astype('f4'),
        y=rng.random((T,N,4)).astype('f4'),
        rotor_angles_deg=np.linspace(0,45,T),
        case_id=f'gpu_{seed}', sequence_len=T,
    )

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.cuda.reset_peak_memory_stats() if device=='cuda' else None
mem_before = torch.cuda.memory_allocated() if device=='cuda' else 0

for i in range(100):
    batch = collate_temporal_batch([_make(seed=i*4+j) for j in range(4)])
    x = torch.tensor(batch['x']).to(device)
    y = torch.tensor(batch['y']).to(device)
    del x, y, batch
    if device=='cuda':
        torch.cuda.empty_cache()

mem_after = torch.cuda.memory_allocated() if device=='cuda' else 0
delta_mb = (mem_after - mem_before) / (1024*1024)
print(json.dumps({'device': device, 'delta_mb': delta_mb, 'mem_before_mb': mem_before/1024/1024, 'mem_after_mb': mem_after/1024/1024}))
"""
    result = subprocess.run(
        ["docker", "exec", "physicsnemo", "python", "-c", script],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, f"Docker script failed:\n{result.stderr}"

    # Parse JSON from last non-empty line
    output_lines = [l for l in result.stdout.strip().splitlines() if l.strip()]
    assert output_lines, f"No output from Docker script.\nstderr: {result.stderr}"
    data = json.loads(output_lines[-1])

    delta_mb = data["delta_mb"]
    assert delta_mb < 200.0, (
        f"GPU memory increased by {delta_mb:.1f} MB over 100 batches on {data['device']}. "
        "Possible memory leak. Check tensor lifecycle in DataLoader batches."
    )
