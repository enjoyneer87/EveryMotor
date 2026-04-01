#!/usr/bin/env python3
"""Validate submodule branch/commit state before running server jobs."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], cwd: Path) -> str:
    proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{proc.stderr.strip()}")
    return proc.stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=".", help="EveryMotor repository root")
    parser.add_argument("--submodule", default="eMach", help="Submodule path")
    parser.add_argument(
        "--expected-branch",
        default="devVeriACLoss",
        help="Branch that submodule must track",
    )
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    submodule_path = repo / args.submodule
    ok = True

    print(f"Repo: {repo}")
    print(f"Submodule: {submodule_path}")

    if not (repo / ".gitmodules").exists():
        print("[FAIL] .gitmodules not found")
        return 2

    if not submodule_path.exists():
        print("[FAIL] Submodule path does not exist")
        return 2

    try:
        status = run(["git", "submodule", "status", "--", args.submodule], repo)
        print(f"submodule status: {status}")

        if status.startswith("-"):
            print("[FAIL] Submodule is not initialized")
            ok = False

        branch = run(["git", "rev-parse", "--abbrev-ref", "HEAD"], submodule_path)
        commit = run(["git", "rev-parse", "--short", "HEAD"], submodule_path)
        print(f"HEAD branch: {branch}")
        print(f"HEAD commit: {commit}")

        if branch == "HEAD":
            print("[FAIL] Submodule is in detached HEAD")
            ok = False

        if branch != args.expected_branch:
            print(f"[FAIL] Expected branch '{args.expected_branch}', got '{branch}'")
            ok = False
        else:
            print(f"[OK] Expected branch matched: {args.expected_branch}")

        tracking = run(
            ["git", "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"],
            submodule_path,
        )
        print(f"upstream: {tracking}")
    except Exception as exc:
        print(f"[FAIL] {exc}")
        return 2

    if ok:
        print("\nPASS: submodule branch contract satisfied")
        return 0

    print("\nFAIL: submodule branch contract violated")
    return 1


if __name__ == "__main__":
    sys.exit(main())
