#!/usr/bin/env python3
"""Produce a lightweight operations status snapshot as JSON."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from runtime_paths import get_runtime_paths


def run_git(repo: Path, args: list[str]) -> str:
    try:
        proc = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
        if proc.returncode != 0:
            return ""
        return proc.stdout.strip()
    except Exception:
        return ""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=".")
    parser.add_argument("--out", help="Optional output JSON path")
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    paths = get_runtime_paths()

    host_data = paths["host_data"]
    doe_data = paths["doe_data"]

    ckpts = {
        "fno": host_data / "doe_fno_ckpt.pt",
        "gino": host_data / "doe_gino_ckpt.pt",
        "mgn": host_data / "doe_meshgraphnet_ckpt.pt",
        "rnn": host_data / "doe_rnn_ckpt.pt",
    }

    git_branch = run_git(repo, ["rev-parse", "--abbrev-ref", "HEAD"])
    git_head = run_git(repo, ["rev-parse", "--short", "HEAD"])
    submodule_status = run_git(repo, ["submodule", "status", "--", "eMach"])

    snapshot = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "repo": str(repo),
        "git": {
            "branch": git_branch,
            "head": git_head,
            "submodule_eMach": submodule_status,
        },
        "runtime": {
            "host_data": {"path": str(host_data), "exists": host_data.exists()},
            "doe_data": {"path": str(doe_data), "exists": doe_data.exists()},
            "manifest": {
                "path": str(doe_data / "doe_manifest.json"),
                "exists": (doe_data / "doe_manifest.json").exists(),
            },
        },
        "checkpoints": {
            name: {"path": str(path), "exists": path.exists()} for name, path in ckpts.items()
        },
        "commands": {
            "python": shutil.which("python") or "",
            "docker": shutil.which("docker") or "",
            "pwsh": shutil.which("pwsh") or "",
            "powershell": shutil.which("powershell") or "",
        },
        "scripts": {
            "sync_notion_fields": (repo / "sync_notion_fields.ps1").exists(),
            "notion_task_update": (repo / "notion_task_update.ps1").exists(),
            "notion_pick_workitem": (repo / "notion_pick_workitem.ps1").exists(),
            "overnight_agent_cycle": (repo / "overnight_agent_cycle.ps1").exists(),
        },
    }

    text = json.dumps(snapshot, indent=2)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")

    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
