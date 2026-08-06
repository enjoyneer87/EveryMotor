"""Assemble the R8 v2 dataset: re-label the fixed-excitation 240 + ingest the 80 pilot cases.

Two operations, both non-destructive to existing data dirs:

1. **Re-label** (not re-solve): section 26 proved every pre-fix case actually ran at
   650.538 A peak regardless of its nominal PeakCurrent. So the existing 240 cases are a
   valid single-current slice -- their manifest entries just carried the wrong label.
   The v2 manifest records `PeakCurrent = 650.538`, keeps the original nominal value as
   `PeakCurrent_nominal_unwired` for provenance, and flags `excitation_wired = 0`.
2. **Ingest** the 80 pilot cases (indices 240..319) by copying their h5 into the v2 dir,
   with `excitation_wired = 1` and `source_geometry_index` preserved so the split builder
   can enforce a current-holdout.

The v2 dir gets its own case dirs so `backup/doe_data*` are never mutated -- the legacy
gate and every published scorecard stay reproducible byte-for-byte.

    python results/logs/doe_build_v2.py            # writes backup/doe_data_v2
    python results/logs/doe_build_v2.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ANCHOR_IPK = 650.538238691624          # section 26: template 460 A RMS -> peak

# Defaults are the host-native layout this was first run on. They are options rather
# than constants so the assembly can be re-run inside the training container (which sees
# the repo at /workspace/app and cannot see the solve host's DOE_CurrentAxis dir at all).
# The pilot h5 are already ingested under the v2 dir, so a re-run only rewrites the
# manifest; --pilot-manifest accepts the committed copy at
# results/logs/doe_current_pilot_manifest.json.
DEFAULT_REPO = Path("D:/KDH/NvidiaNemo")
DEFAULT_PILOT_MANIFEST = Path("D:/KDH/Sim_4SolverX/DOE_CurrentAxis/doe_manifest.json")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--repo", type=Path, default=DEFAULT_REPO,
                    help="repo root; stored h5 paths are made relative to it")
    ap.add_argument("--pilot-manifest", type=Path, default=DEFAULT_PILOT_MANIFEST)
    ap.add_argument("--skip-copy", action="store_true",
                    help="manifest-only rebuild; the pilot h5 are already ingested")
    args = ap.parse_args()

    global REPO, BASE, OUT
    REPO = args.repo
    BASE = REPO / "backup/doe_data_240"    # the 240-case fixed-excitation manifest
    OUT = REPO / "backup/doe_data_v2"
    sys.path.insert(0, str(REPO))

    base = json.loads((BASE / "doe_manifest.json").read_text(encoding="utf-8"))
    pilot = json.loads(args.pilot_manifest.read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)

    # The 240-case manifest stores the SOLVE HOST's absolute paths
    # (D:/KDH/Sim_4SolverX/DOE_TrainingData/...), which no longer exist; the files were
    # ingested into backup/doe_data{,_120,_240}. eval.doe_dataset.resolve_h5_path handles
    # that at load time by falling back to <data_dir>/case_XXXX/postproc/<basename>, but
    # only for the data_dir it is given -- and the doe240 set legitimately spans two dirs
    # (cases 40-119 physically live under doe_data_120). So resolve here, once, and store
    # paths that actually exist. A case whose files cannot be found is a hard error: a
    # silently short dataset is exactly the failure class this campaign keeps auditing for.
    #
    # The stored path must be REPO-RELATIVE with forward slashes. The first version of
    # this builder wrote host-absolute paths (D:\KDH\NvidiaNemo\backup\...), which the
    # training container -- it sees the repo as /workspace/app and nothing else -- could
    # not resolve for a single one of the 240 legacy cases (0/240; the pilot 80 resolved
    # only because they physically live under the v2 dir). resolve_h5_path tries
    # Path(recorded) first, and both the container and a host-native run have the repo
    # root as cwd, so a relative path resolves on both. Regression-tested by
    # results/logs/_v2_path_probe.py, which must report 320/320.
    from eval.doe_dataset import resolve_h5_path

    search_dirs = [REPO / "backup/doe_data_240", REPO / "backup/doe_data_120",
                   REPO / "backup/doe_data"]

    def rel(p: Path) -> str:
        """Repo-relative POSIX path, so it resolves under any repo mount point."""
        return Path(p).resolve().relative_to(REPO.resolve()).as_posix()

    def resolve(recorded: str, case_index: int) -> str:
        for d in search_dirs:
            got = resolve_h5_path(recorded, d, case_index)
            if got is not None:
                return rel(got)
        raise FileNotFoundError(f"case {case_index}: cannot locate {recorded}")

    cases = []
    # ---- 1) re-label the existing 240 (files stay where they are; paths resolved) ----
    for c in base["cases"]:
        elec = dict(c.get("electrical", {}))
        nominal = elec.get("PeakCurrent")
        elec["PeakCurrent_nominal_unwired"] = nominal
        elec["PeakCurrent"] = ANCHOR_IPK
        elec["CurrentDefinition"] = 1
        elec["excitation_wired"] = 0
        idx = int(c["index"])
        cases.append({
            "index": idx,
            "source_geometry_index": idx,
            "geometry": c.get("geometry", {}),
            "electrical": elec,
            "h5_paths": [resolve(p, idx) for p in c["h5_paths"]],
            "solve_time_s": c.get("solve_time_s"),
        })

    # ---- 2) ingest the pilot at offset 240 ----
    offset = 240
    for c in pilot["cases"]:
        k = offset + int(c["index"])
        dst_pp = OUT / f"case_{k:04d}" / "postproc"
        h5_dst = []
        for src in c["h5_paths"]:
            src_p = Path(src)
            # The pilot manifest records Windows paths; Path(...).name does not split
            # backslashes on Linux, so take the basename OS-agnostically.
            dst = dst_pp / str(src).replace("\\", "/").split("/")[-1]
            if not args.dry_run and not args.skip_copy:
                dst_pp.mkdir(parents=True, exist_ok=True)
                if not dst.exists():
                    shutil.copy2(src_p, dst)
            if args.skip_copy and not dst.exists():
                raise FileNotFoundError(
                    f"case {k}: --skip-copy but {dst} is missing -- re-run without it")
            h5_dst.append(rel(dst))
        entry = {
            "index": k,
            "source_geometry_index": int(c["source_geometry_index"]),
            "geometry": c.get("geometry", {}),
            "electrical": dict(c.get("electrical", {})),
            "h5_paths": h5_dst,
            "solve_time_s": c.get("solve_time_s"),
        }
        cases.append(entry)
        if not args.dry_run:
            (OUT / f"case_{k:04d}" / "meta.json").write_text(
                json.dumps(entry, indent=2), encoding="utf-8")

    manifest = {
        "n_cases": len(cases),
        "campaign": "r8_v2_current_axis",
        "anchor_ipk": ANCHOR_IPK,
        "note": ("240 fixed-excitation cases re-labelled to their true excitation "
                 "(650.538 A peak, section 26) + 80 pilot cases at 325.3/162.6 A. "
                 "The 240 entries reference the original doe_data_240 h5 files, so "
                 "those dirs stay untouched and every legacy scorecard remains "
                 "reproducible."),
        "cases": cases,
    }
    if not args.dry_run:
        (OUT / "doe_manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")

    levels = {}
    for c in cases:
        levels[round(c["electrical"]["PeakCurrent"], 1)] = \
            levels.get(round(c["electrical"]["PeakCurrent"], 1), 0) + 1
    print(f"v2 cases: {len(cases)}  levels: {levels}")
    print(f"wired=1: {sum(1 for c in cases if c['electrical'].get('excitation_wired'))}, "
          f"wired=0: {sum(1 for c in cases if not c['electrical'].get('excitation_wired'))}")
    print(("DRY RUN, nothing written" if args.dry_run else f"wrote {OUT}/doe_manifest.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
