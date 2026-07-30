"""Build a case-holdout split for an expanded DOE dir, holding test/val IDENTICAL to
doe40 so |B| generalization is measured on the same 6 geometries across all scales.

Env: DOE_DATA (dir with doe_manifest.json), DOE_SPLIT_OUT (json path).
Run in the container (needs eval.case_split): PYTHONPATH=/workspace/app python this.
"""
import json
import os
from pathlib import Path
from eval.case_split import manifest_digest

DATA = Path(os.environ["DOE_DATA"])
OUT = Path(os.environ["DOE_SPLIT_OUT"])
TEST = [4, 7, 18, 32, 37, 39]   # identical to doe40_case_split
VAL = [5, 16, 25, 28]

man = json.loads((DATA / "doe_manifest.json").read_text(encoding="utf-8"))
digest = manifest_digest(man)
all_idx = sorted(int(c["index"]) for c in man["cases"])
held = set(TEST) | set(VAL)
train = [i for i in all_idx if i not in held]
split = {
    "version": "case_split/v1", "seed": 42, "doe_digest": digest,
    "sizes": {"train": len(train), "val": len(VAL), "test": len(TEST)},
    "granularity": "case",
    "note": f"DOE expansion {len(all_idx)} cases. Test/val identical to doe40 so |B| "
            "generalization is measured on the same 6 geometries; only train grows.",
    "train": train, "val": VAL, "test": TEST,
}
OUT.write_text(json.dumps(split, indent=2), encoding="utf-8")
print(f"digest {digest} | n_cases {len(all_idx)} | train {len(train)} val {len(VAL)} test {len(TEST)}")
print(f"wrote {OUT}")
