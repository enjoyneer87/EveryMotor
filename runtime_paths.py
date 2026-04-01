#!/usr/bin/env python3
"""Shared runtime path contract for EveryMotor scripts."""

from __future__ import annotations

import os
from pathlib import Path


def _first_env(keys: list[str], default: str) -> str:
    for key in keys:
        val = os.getenv(key, "").strip()
        if val:
            return val
    return default


def get_runtime_paths() -> dict[str, Path]:
    host_data = Path(
        _first_env(
            ["HOST_DATA_DIR", "EM_HOST_DATA", "EVERYMOTOR_HOST_DATA"],
            "/workspace/host_data",
        )
    )
    doe_data = Path(
        _first_env(
            ["DOE_DATA_DIR", "EM_DOE_DATA", "EVERYMOTOR_DOE_DATA"],
            str(host_data / "doe_data"),
        )
    )
    mso_root = Path(
        _first_env(
            ["MSO_ROOT", "EM_MSO_ROOT", "EVERYMOTOR_MSO_ROOT"],
            "/workspace/multiscale-pde-operators",
        )
    )

    return {
        "host_data": host_data,
        "doe_data": doe_data,
        "mso_root": mso_root,
    }
