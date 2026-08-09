"""Precompute the R7-A prior cache for a dataset.

The graph builder computes priors on demand (~1.3 s/graph cold), which would
add hours to a training launch; this driver pays that cost once, writing
``results/prior_cache/<dataset>/case_XXXX.npz``. Safe to re-run — cached steps
are skipped. Single-process on purpose: it is meant to run alongside a GPU
training run without starving its dataloader.

    python -m fem_warmstart.build_prior_cache --data-dir backup/doe_data_240
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np  # noqa: F401  (keeps the import cost out of the loop)
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.doe_dataset import load_doe_cases                 # noqa: E402
from fem_warmstart.prior import nodal_prior_features        # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", type=Path, default=Path("backup/doe_data_240"))
    ap.add_argument("--step-stride", type=int, default=1)
    # An assembled dataset (v3 = v2's 320 verbatim + 160 new) reuses the caches of the
    # dirs its h5 files physically live in -- prior.py keys the cache on the h5 path, not
    # on --data-dir. Without a range, a v3 run re-parses 320 already-cached cases from
    # disk for nothing (~30 min). Half-open [from, to), defaults to the whole manifest.
    ap.add_argument("--cases-from", type=int, default=0)
    ap.add_argument("--cases-to", type=int, default=None)
    args = ap.parse_args()

    manifest = json.loads((args.data_dir / "doe_manifest.json").read_text(encoding="utf-8"))
    n_cases = int(manifest["n_cases"])
    lo = max(0, args.cases_from)
    hi = n_cases if args.cases_to is None else min(n_cases, args.cases_to)
    t0 = time.time()
    done = 0
    for ci in range(lo, hi):
        try:
            rec = load_doe_cases(manifest, args.data_dir, case_indices=[ci]).records[0]
        except Exception as exc:                            # noqa: BLE001
            print(f"case {ci}: load failed ({exc}); skipped", flush=True)
            continue
        tc = time.time()
        for si in range(0, len(rec.samples), max(1, args.step_stride)):
            nodal_prior_features(rec, rec.samples[si])
            done += 1
        print(f"case {ci:3d}/{n_cases}  {len(rec.samples)} steps  "
              f"{time.time()-tc:5.1f}s  (total {done} solves, {time.time()-t0:6.0f}s)",
              flush=True)
    print(f"PRIOR_CACHE_DONE {done} solves in {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
