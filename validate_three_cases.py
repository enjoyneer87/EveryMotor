#!/usr/bin/env python3
"""Run 3-case validation pipeline for EveryMotor inference outputs.

This script prepares and executes infer_all_steps.py for three case indices,
then writes a machine-readable summary JSON for Notion evidence updates.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from runtime_paths import get_runtime_paths


def pick_case_indices(data_dir: Path, count: int) -> list[int]:
    manifest_path = data_dir / "doe_manifest.json"
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    case_ids: list[int] = []
    for case in manifest.get("cases", []):
        idx = case.get("index")
        if isinstance(idx, int):
            case_ids.append(idx)

    case_ids = sorted(set(case_ids))
    return case_ids[:count]


def run_one(case_idx: int, infer_script: Path, data_dir: Path, out_file: Path) -> dict:
    cmd = [
        sys.executable,
        str(infer_script),
        "--data-dir",
        str(data_dir),
        "--case-idx",
        str(case_idx),
        "--out",
        str(out_file),
    ]

    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    dt = round(time.time() - t0, 3)

    return {
        "case_idx": case_idx,
        "command": cmd,
        "returncode": proc.returncode,
        "duration_sec": dt,
        "out_file": str(out_file),
        "stdout_tail": proc.stdout[-2000:],
        "stderr_tail": proc.stderr[-2000:],
        "ok": proc.returncode == 0 and out_file.exists(),
    }


def main() -> int:
    paths = get_runtime_paths()

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(paths["doe_data"]))
    parser.add_argument("--infer-script", default="infer_all_steps.py")
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--out-dir", default=str(paths["host_data"] / "validation_3cases"))
    parser.add_argument("--summary", default=str(paths["host_data"] / "validation_3cases" / "summary.json"))
    parser.add_argument("--case-indices", nargs="*", type=int)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    infer_script = Path(args.infer_script)
    out_dir = Path(args.out_dir)
    summary_path = Path(args.summary)

    out_dir.mkdir(parents=True, exist_ok=True)

    if args.case_indices:
        case_indices = args.case_indices
    else:
        case_indices = pick_case_indices(data_dir, args.count)

    if not case_indices:
        print("No case indices available for validation.")
        return 2

    results = []
    for case_idx in case_indices:
        out_file = out_dir / f"case_{case_idx:04d}_allsteps.npz"
        print(f"[RUN] case_idx={case_idx} -> {out_file}")
        result = run_one(case_idx=case_idx, infer_script=infer_script, data_dir=data_dir, out_file=out_file)
        results.append(result)
        state = "OK" if result["ok"] else "FAIL"
        print(f"[{state}] case_idx={case_idx} returncode={result['returncode']} duration={result['duration_sec']}s")

    summary = {
        "ok": all(r["ok"] for r in results),
        "count": len(results),
        "results": results,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Summary: {summary_path}")

    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
