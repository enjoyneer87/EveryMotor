#!/usr/bin/env python3
"""Validate runtime path and checkpoint contract for EveryMotor server jobs."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def resolve_path(raw: str) -> Path:
    expanded = os.path.expandvars(raw)
    return Path(expanded).resolve()


def check_path(name: str, path: Path, must_exist: bool) -> bool:
    exists = path.exists()
    state = "OK" if (exists or not must_exist) else "FAIL"
    print(f"[{state}] {name}: {path} (exists={exists})")
    return exists or not must_exist


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host-data", default="/workspace/host_data")
    parser.add_argument("--doe-data", default="/workspace/host_data/doe_data")
    parser.add_argument("--mso-root", default="/workspace/multiscale-pde-operators")
    parser.add_argument("--require-checkpoints", action="store_true")
    args = parser.parse_args()

    ok = True
    host_data = resolve_path(args.host_data)
    doe_data = resolve_path(args.doe_data)
    mso_root = resolve_path(args.mso_root)

    print("Runtime contract check")
    print("=" * 40)

    ok &= check_path("HOST_DATA", host_data, must_exist=True)
    ok &= check_path("DOE_DATA", doe_data, must_exist=True)
    ok &= check_path("MSO_ROOT", mso_root, must_exist=True)

    manifest = doe_data / "doe_manifest.json"
    ok &= check_path("DOE_MANIFEST", manifest, must_exist=True)

    ckpts = [
        host_data / "doe_fno_ckpt.pt",
        host_data / "doe_gino_ckpt.pt",
        host_data / "doe_meshgraphnet_ckpt.pt",
        host_data / "doe_rnn_ckpt.pt",
    ]
    print("\nCheckpoint contract")
    print("-" * 40)
    for ckpt in ckpts:
        ok &= check_path(ckpt.name, ckpt, must_exist=args.require_checkpoints)

    print("\nPython package quick check")
    print("-" * 40)
    packages = ["torch", "h5py", "scipy", "flask", "physicsnemo", "neuralop"]
    for pkg in packages:
        try:
            __import__(pkg)
            print(f"[OK] import {pkg}")
        except Exception as exc:
            print(f"[WARN] import {pkg} failed: {exc}")
            if pkg in ("torch", "h5py", "scipy"):
                ok = False

    if ok:
        print("\nPASS: runtime contract satisfied")
        return 0

    print("\nFAIL: runtime contract violated")
    return 1


if __name__ == "__main__":
    sys.exit(main())
