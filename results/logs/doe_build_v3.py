"""Assemble the R9 v3 dataset: v2's 320 cases + the 160-case Ext3 expansion.

Non-destructive, same contract as doe_build_v2.py: existing data dirs are never
mutated, stored h5 paths are REPO-RELATIVE POSIX so they resolve identically on the
host and inside the training container (the v2 path-gate lesson, section 27/29).

  - indices   0..319: copied VERBATIM from backup/doe_data_v2/doe_manifest.json
                      (paths already repo-relative and container-verified)
  - indices 320..479: Ext3 case k -> v3 index 320+k; h5 copied into
                      backup/doe_data_v3/case_XXXX/postproc/. source_geometry_index
                      is offset by 240 into a fresh geometry-id space (Ext3 rows
                      0..119 -> geometry ids 240..359) so the split builder can
                      never collide them with the original 0..239 geometry ids.

Split v3 (written alongside): v2's val/test UNCHANGED (test/val held fixed across
scales, section 20 principle); every new case goes to train -> 450/12/18.

  python results/logs/doe_build_v3.py [--dry-run]     # container: --repo /workspace/app --ext3 /workspace/ext3
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

DEFAULT_REPO = Path("D:/KDH/NvidiaNemo")
DEFAULT_EXT3 = Path("D:/KDH/Sim_4SolverX/DOE_Ext3")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    ap.add_argument("--ext3", type=Path, default=DEFAULT_EXT3)
    args = ap.parse_args()

    repo = args.repo
    sys.path.insert(0, str(repo))
    v2 = json.loads((repo / "backup/doe_data_v2/doe_manifest.json").read_text(encoding="utf-8"))
    ext3 = json.loads((args.ext3 / "doe_manifest.json").read_text(encoding="utf-8"))
    if ext3.get("missing"):
        raise SystemExit(f"Ext3 manifest lists missing cases {ext3['missing']} -- "
                         "finish generation (or accept explicitly) before assembling v3")
    out = repo / "backup/doe_data_v3"
    out.mkdir(parents=True, exist_ok=True)

    def rel(p: Path) -> str:
        return Path(p).resolve().relative_to(repo.resolve()).as_posix()

    cases = [dict(c) for c in v2["cases"]]                 # 0..319 verbatim
    assert len(cases) == 320

    for c in ext3["cases"]:
        k = 320 + int(c["index"])
        dst_pp = out / f"case_{k:04d}" / "postproc"
        h5_dst = []
        for src in c["h5_paths"]:
            basename = str(src).replace("\\", "/").split("/")[-1]
            dst = dst_pp / basename
            if not args.dry_run:
                dst_pp.mkdir(parents=True, exist_ok=True)
                if not dst.exists():
                    shutil.copy2(src, dst)
            h5_dst.append(rel(dst))
        entry = {
            "index": k,
            # fresh geometry-id space: Ext3 LHS row g -> 240+g, disjoint from 0..239
            "source_geometry_index": 240 + int(c["source_geometry_index"]),
            "geometry": c.get("geometry", {}),
            "electrical": dict(c.get("electrical", {})),
            "h5_paths": h5_dst,
            "solve_time_s": c.get("solve_time_s"),
        }
        cases.append(entry)
        if not args.dry_run:
            (out / f"case_{k:04d}" / "meta.json").write_text(
                json.dumps(entry, indent=2), encoding="utf-8")

    manifest = {
        "n_cases": len(cases),
        "campaign": "r9_v3_480",
        "anchor_ipk": v2.get("anchor_ipk"),
        "note": ("v2's 320 cases verbatim (repo-relative paths, container-verified) + "
                 "Ext3 batch (120 new geometries @650.538 A + 20 of them x {325.269, "
                 "162.635} A, seed 46, section 30 design). doe_data_v2 and earlier dirs "
                 "stay untouched."),
        "cases": cases,
    }
    if not args.dry_run:
        (out / "doe_manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")

    # ---- split v3: v2 val/test frozen, all new cases -> train --------------------
    from eval.case_split import manifest_digest
    v2_split = json.loads((repo / "eval/splits/doe_v2_case_split.json").read_text(encoding="utf-8"))
    split = dict(v2_split)
    split["version"] = "case_split/v3"
    split["train"] = sorted(set(int(c) for c in v2_split["train"]) | set(range(320, 480)))
    split["doe_digest"] = manifest_digest(manifest)
    split["sizes"] = {"train": len(split["train"]), "val": len(split["val"]),
                      "test": len(split["test"]), "test_legacy": 6, "test_current": 12}
    split["note"] = ("v3: v2 val/test UNCHANGED (fixed across scales, section 20); the "
                     "160 Ext3 cases (geometry ids 240..359, all-new geometries) train "
                     "only. " + v2_split.get("note", ""))
    if not args.dry_run:
        (repo / "eval/splits/doe_v3_case_split.json").write_text(
            json.dumps(split, indent=1), encoding="utf-8")

    lv = {}
    for c in cases:
        lv[round(c["electrical"]["PeakCurrent"], 1)] = lv.get(round(c["electrical"]["PeakCurrent"], 1), 0) + 1
    print(f"v3 cases: {len(cases)}  levels: {lv}")
    print(f"split v3: train {len(split['train'])} / val {len(split['val'])} / test {len(split['test'])}  "
          f"digest {split['doe_digest']}")
    print("DRY RUN, nothing written" if args.dry_run else f"wrote {out}/doe_manifest.json + eval/splits/doe_v3_case_split.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
