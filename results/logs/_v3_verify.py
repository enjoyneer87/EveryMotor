"""v3 assembly verification: paths resolve, and the rev-1 zero-current defect is gone.

The split digest cannot catch a re-solve: manifest_digest hashes only case index +
geometry + electrical VALUES, so the rev-1 (broken, conductor dimensions never
recomputed after the geometry change) and rev-2 (fixed) batches share digest
e249eb764ebf8865 despite carrying different physics. Data identity is therefore
established by the ampere-turns gate and by this probe, not by the digest -- worth
remembering the next time a digest match is mistaken for "same data".
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/workspace/app")
from eval.doe_dataset import load_doe_cases, resolve_h5_path   # noqa: E402

DATA = Path("backup/doe_data_v3")
# v3 index = 320 + ext3 index; 1/14/17 and 132/133 were rev-1 zero-current casualties
PROBE = [321, 334, 337, 452, 453, 320, 479]

man = json.loads((DATA / "doe_manifest.json").read_text(encoding="utf-8"))
bad = [int(c["index"]) for c in man["cases"]
       if any(resolve_h5_path(p, DATA, int(c["index"])) is None for p in c["h5_paths"])]
print(f"path gate: unresolved {len(bad)}/480")
print("V3_PATH_GATE_" + ("PASS" if not bad else "FAIL"))

zero = []
for ci in PROBE:
    rec = load_doe_cases(man, DATA, case_indices=[ci]).records[0]
    j = np.asarray(rec.samples[0].fields["j"])
    ipk = float(rec.condition.get("PeakCurrent", float("nan")))
    jmax = float(np.abs(j).max())
    if jmax == 0.0:
        zero.append(ci)
    print(f"  case {ci:4d} (ext3 {ci - 320:3d}, {ipk:7.1f} A): |j|max {jmax:8.4f}  "
          f"{'ZERO-CURRENT' if jmax == 0.0 else 'OK'}")
print("V3_CURRENT_PROBE_" + ("PASS" if not zero else f"FAIL {zero}"))
sys.exit(0 if (not bad and not zero) else 1)
