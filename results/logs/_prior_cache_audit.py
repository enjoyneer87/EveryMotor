"""Is the v3 prior cache actually consistent with the v3 meshes on disk?

The 160 doe_data_v3 cache entries were written by a process this session did not run
(timestamps sit after the rev-2 solves but before this session's v3 re-assembly), so
their provenance is inferred rather than known. The read path only validates
`arr.shape == (n_nodes, 3)`, which a stale entry from a different mesh of the same node
count would pass silently -- and a wrong prior would poison a 48 h training run in a way
no downstream metric attributes back to the cache.

So: recompute the prior from scratch (PRIOR_CACHE_DISABLE) for a sample of cases and
compare against what the cache serves. Bit-level agreement is expected -- the solve is a
deterministic sparse direct factorization on fixed matrices.

    PRIOR_CACHE_DISABLE= python results/logs/_prior_cache_audit.py
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/workspace/app")
from eval.doe_dataset import load_doe_cases                      # noqa: E402
from fem_warmstart.prior import compute_prior, nodal_prior_features, PRIOR_FEATURE_NAMES  # noqa: E402

DATA = Path("backup/doe_data_v3")
# spread across the new batch: first, a former zero-current casualty, factorial variants, last
CASES = [320, 321, 337, 452, 453, 479]
STEPS = [0, 22, 44]

man = json.loads((DATA / "doe_manifest.json").read_text(encoding="utf-8"))
worst = 0.0
bad = []
for ci in CASES:
    rec = load_doe_cases(man, DATA, case_indices=[ci]).records[0]
    for si in STEPS:
        sample = rec.samples[si]
        cached = nodal_prior_features(rec, sample, cache=True)          # cache path
        fresh_arr = compute_prior(rec, si)                              # no cache, recompute
        fresh = {n: np.ascontiguousarray(fresh_arr[:, i])
                 for i, n in enumerate(PRIOR_FEATURE_NAMES)}
        for name in PRIOR_FEATURE_NAMES:
            c, f = np.asarray(cached[name], dtype=np.float64), fresh[name]
            if c.shape != f.shape:
                bad.append((ci, si, name, f"shape {c.shape} vs {f.shape}"))
                continue
            scale = max(float(np.abs(f).max()), 1e-30)
            rel = float(np.abs(c - f).max()) / scale
            worst = max(worst, rel)
            if rel > 1e-6:                        # float32 cache storage -> ~1e-7 expected
                bad.append((ci, si, name, f"rel {rel:.3e}"))
    print(f"case {ci:4d}: checked steps {STEPS}, worst rel so far {worst:.3e}", flush=True)

print(f"max relative deviation cache-vs-recompute: {worst:.3e}")
for b in bad[:10]:
    print("  MISMATCH", b)
print("PRIOR_CACHE_AUDIT_" + ("PASS" if not bad else "FAIL"))
sys.exit(0 if not bad else 1)
