#!/usr/bin/env python3
"""Lightweight subprocess bridge with JSON input/output contract.

This module supports XENV tasks:
- XENV-2: subprocess bridge
- XENV-3: JSON I/O contract
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class CommandResult:
    ok: bool
    command: list[str]
    cwd: str
    timeout_sec: int
    returncode: int
    stdout: str
    stderr: str
    start_utc: str
    end_utc: str
    duration_sec: float


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_command(command: list[str], cwd: str | None = None, timeout_sec: int = 300) -> CommandResult:
    start_time = time.time()
    start_utc = now_utc_iso()
    proc = subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout_sec,
        shell=False,
    )
    end_utc = now_utc_iso()
    return CommandResult(
        ok=proc.returncode == 0,
        command=command,
        cwd=str(Path(cwd).resolve()) if cwd else str(Path.cwd()),
        timeout_sec=timeout_sec,
        returncode=proc.returncode,
        stdout=proc.stdout,
        stderr=proc.stderr,
        start_utc=start_utc,
        end_utc=end_utc,
        duration_sec=round(time.time() - start_time, 3),
    )


def run_from_payload(payload: dict[str, Any]) -> CommandResult:
    command = payload.get("command")
    if not isinstance(command, list) or not command or not all(isinstance(x, str) for x in command):
        raise ValueError("payload.command must be a non-empty string list")

    cwd = payload.get("cwd")
    timeout_sec = int(payload.get("timeout_sec", 300))
    return run_command(command=command, cwd=cwd, timeout_sec=timeout_sec)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload", required=True, help="Path to JSON payload")
    parser.add_argument("--out", help="Optional path to write JSON result")
    args = parser.parse_args()

    payload_path = Path(args.payload)
    if not payload_path.exists():
        print(json.dumps({"error": f"payload not found: {payload_path}"}))
        return 2

    with payload_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    try:
        result = run_from_payload(payload)
        result_json = json.dumps(asdict(result), ensure_ascii=False, indent=2)
    except subprocess.TimeoutExpired as exc:
        error_result = {
            "ok": False,
            "command": payload.get("command", []),
            "cwd": str(Path(payload.get("cwd", ".")).resolve()),
            "timeout_sec": int(payload.get("timeout_sec", 300)),
            "returncode": -1,
            "stdout": "",
            "stderr": str(exc),
            "start_utc": now_utc_iso(),
            "end_utc": now_utc_iso(),
            "duration_sec": 0.0,
            "error": "timeout",
        }
        result_json = json.dumps(
            error_result,
            ensure_ascii=False,
            indent=2,
        )
    except Exception as exc:
        error_result = {
            "ok": False,
            "command": payload.get("command", []) if isinstance(payload, dict) else [],
            "cwd": str(Path(".").resolve()),
            "timeout_sec": int(payload.get("timeout_sec", 300)) if isinstance(payload, dict) else 300,
            "returncode": -1,
            "stdout": "",
            "stderr": str(exc),
            "start_utc": now_utc_iso(),
            "end_utc": now_utc_iso(),
            "duration_sec": 0.0,
            "error": str(exc),
        }
        result_json = json.dumps(error_result, ensure_ascii=False, indent=2)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(result_json + os.linesep, encoding="utf-8")

    print(result_json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
