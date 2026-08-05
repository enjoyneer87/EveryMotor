"""Build split v2: geometry-holdout x current-holdout for the R8 dataset.

The campaign's whole comparability rests on one rule -- the same six geometries have
never been trained on -- and R8 adds a second axis that needs the same treatment, or
"generalizes to new current" would silently mean "interpolates a current it saw on
this very geometry".

    TEST_GEOM = [4, 7, 18, 32, 37, 39]   (unchanged since doe40)
    VAL_GEOM  = [5, 16, 25, 28]          (unchanged)

    test : every case whose source geometry is in TEST_GEOM, at ANY current level
           -> splits into two reported groups:
              test_legacy  = the 650.538 A slice (identical samples to the legacy gate)
              test_current = the new 325.3 / 162.6 A slices (unseen geometry AND
                             unseen-for-that-geometry current)
    val  : same construction on VAL_GEOM
    train: everything else -- i.e. the 230 train geometries at 650.538 A plus their
           new current levels, which is where the current-dependence signal comes from

Because the held-out geometries contribute no case at any current, a model cannot
learn "this geometry's response to current" and be credited for current
generalization. That is stricter than holding out only new levels, and it is the
version worth publishing.

    python results/logs/doe_make_split_v2.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, "D:/KDH/NvidiaNemo")

from eval.case_split import manifest_digest        # noqa: E402

DATA = Path(os.environ.get("DOE_DATA", "D:/KDH/NvidiaNemo/backup/doe_data_v2"))
OUT = Path(os.environ.get("DOE_SPLIT_OUT", "D:/KDH/NvidiaNemo/eval/splits/doe_v2_case_split.json"))
TEST_GEOM = [4, 7, 18, 32, 37, 39]
VAL_GEOM = [5, 16, 25, 28]
LEGACY_IPK = 650.538238691624


def main() -> int:
    man = json.loads((DATA / "doe_manifest.json").read_text(encoding="utf-8"))
    digest = manifest_digest(man)

    by_index = {int(c["index"]): c for c in man["cases"]}
    def geom_of(c):
        return int(c.get("source_geometry_index", c["index"]))
    def is_legacy(c):
        return abs(float(c["electrical"]["PeakCurrent"]) - LEGACY_IPK) < 1e-6

    test, test_legacy, test_current = [], [], []
    val, train = [], []
    for idx, c in sorted(by_index.items()):
        g = geom_of(c)
        if g in TEST_GEOM:
            test.append(idx)
            (test_legacy if is_legacy(c) else test_current).append(idx)
        elif g in VAL_GEOM:
            val.append(idx)
        else:
            train.append(idx)

    levels = sorted({round(float(c["electrical"]["PeakCurrent"]), 1) for c in man["cases"]})
    split = {
        "version": "case_split/v2",
        "seed": 42,
        "doe_digest": digest,
        "granularity": "case",
        "holdout": "geometry x current",
        "current_levels_A_peak": levels,
        "test_geometries": TEST_GEOM,
        "val_geometries": VAL_GEOM,
        "sizes": {"train": len(train), "val": len(val), "test": len(test),
                  "test_legacy": len(test_legacy), "test_current": len(test_current)},
        "note": ("Every case of a held-out geometry is held out at EVERY current level, so "
                 "current generalization cannot be credited to having seen that geometry's "
                 "current response. test_legacy is the 650.538 A slice and contains the same "
                 "samples the legacy fixed-excitation gate scores, which keeps the campaign "
                 "comparable; test_current is the new-current evaluation."),
        "train": train, "val": val, "test": test,
        "test_groups": {"legacy_650A": test_legacy, "new_current": test_current},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(split, indent=2), encoding="utf-8")
    print(f"digest {digest}")
    print(f"train {len(train)}  val {len(val)}  test {len(test)} "
          f"(legacy {len(test_legacy)} / new-current {len(test_current)})")
    print(f"levels {levels}")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
