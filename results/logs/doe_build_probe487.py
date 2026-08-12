"""Assemble the eval-only 487.9 A interpolation probe into the repo.

Section 30 generated six cases: the SIX LEGACY TEST GEOMETRIES re-solved at
487.904 A peak (0.75x the legacy excitation). They were never trained on, at a
current level that was never trained on -- so this reads the current axis
*between* the two levels the model did see (325.3 A and 650.5 A) rather than
outside them.

Not a gate. Section 30 registered it as measurement: if pooled torque lands
between the 325.3 A group (2.404%) and the 650.5 A legacy gate (3.270%),
interpolation holds.

Why assemble rather than point at it: the probe lives in D:/KDH/Sim_4SolverX,
and the training container only sees the repo. Same contract as doe_build_v3 --
h5 copied in, paths stored REPO-RELATIVE POSIX so they resolve identically on the
host and in the container.

    python results/logs/doe_build_probe487.py [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

DEFAULT_REPO = Path("D:/KDH/NvidiaNemo")
DEFAULT_SRC = Path("D:/KDH/Sim_4SolverX/DOE_InterpProbe")
LEGACY_TEST = [4, 7, 18, 32, 37, 39]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC)
    args = ap.parse_args()

    repo = args.repo
    sys.path.insert(0, str(repo))
    src = json.loads((args.src / "doe_manifest.json").read_text(encoding="utf-8"))
    out = repo / "backup/doe_data_probe487"
    out.mkdir(parents=True, exist_ok=True)

    def rel(p: Path) -> str:
        return Path(p).resolve().relative_to(repo.resolve()).as_posix()

    cases = []
    for c in src["cases"]:
        k = int(c["index"])
        dst_pp = out / f"case_{k:04d}" / "postproc"
        h5_dst = []
        for s in c["h5_paths"]:
            dst = dst_pp / str(s).replace("\\", "/").split("/")[-1]
            if not args.dry_run:
                dst_pp.mkdir(parents=True, exist_ok=True)
                if not dst.exists():
                    shutil.copy2(s, dst)
            h5_dst.append(rel(dst))
        cases.append({
            "index": k,
            # The probe re-solves the legacy TEST geometries, so keep their identity
            # visible: this is an unseen CURRENT on an unseen GEOMETRY, and the
            # reader should be able to tell which legacy case each row came from.
            "source_geometry_index": LEGACY_TEST[k] if k < len(LEGACY_TEST) else -1,
            "geometry": c.get("geometry", {}),
            "electrical": dict(c.get("electrical", {})),
            "h5_paths": h5_dst,
            "solve_time_s": c.get("solve_time_s"),
        })

    manifest = {
        "n_cases": len(cases),
        "campaign": "interp_probe_487A_eval_only",
        "note": ("six legacy test geometries at 487.904 A peak (0.75x legacy). "
                 "EVAL ONLY -- never train on this. Section 30 registers it as a "
                 "measurement of current-axis interpolation, not as a gate."),
        "cases": cases,
    }
    if not args.dry_run:
        (out / "doe_manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")

    from eval.case_split import manifest_digest
    digest = manifest_digest(manifest)
    # Everything is test. Train/val are empty on purpose: a split that offered
    # these cases for training would be a licence to leak the probe.
    split = {
        "version": "case_split/probe487",
        "seed": 42,
        "doe_digest": digest,
        "granularity": "case",
        "holdout": "geometry x unseen current level",
        "current_levels_A_peak": [487.9],
        "train": [], "val": [], "test": sorted(c["index"] for c in cases),
        "sizes": {"train": 0, "val": 0, "test": len(cases)},
        "note": ("eval-only probe; train/val deliberately empty so this manifest can "
                 "never be used for training"),
    }
    if not args.dry_run:
        (repo / "eval/splits/doe_probe487_case_split.json").write_text(
            json.dumps(split, indent=1), encoding="utf-8")

    ipk = {round(c["electrical"].get("PeakCurrent", float("nan")), 1) for c in cases}
    print(f"probe cases: {len(cases)}  currents: {sorted(ipk)}  digest {digest}")
    print(f"geometries: {[c['source_geometry_index'] for c in cases]}")
    print("DRY RUN, nothing written" if args.dry_run
          else f"wrote {out}/doe_manifest.json + eval/splits/doe_probe487_case_split.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
