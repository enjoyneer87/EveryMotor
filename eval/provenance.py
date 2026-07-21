"""Record where a scorecard came from.

Why
---
AGENTS.md already requires that any result written into a `.md` names the
machine that produced it — hostname and IP — because the stacks differ per
machine (container vs native) and numbers are only comparable when the origin
is known. The scorecards themselves carried none of that: `environment` held
python, platform and numpy, which does not distinguish the container from the
native venv, does not say which commit produced the harness, and does not say
whether the tree was clean at the time.

That gap has already cost this repo once in the neighbouring form: checkpoints
with the same filename held different weights, which is why
`eval/ckpt_identity.py` exists. This is the same fix one level up — the model's
identity was pinned to its hash, and now the *run's* identity is pinned to its
commit, its machine and its command.

What is deliberately not here
-----------------------------
Nothing is inferred or backfilled. Scorecards written before this module keep
their thin `environment` block; a provenance field invented after the fact
would be a guess wearing a record's clothes. Fields that cannot be read on a
given machine come back ``None`` rather than being filled with a plausible
default.
"""

from __future__ import annotations

import getpass
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

# A dirty tree is worth recording, but the file list is a record, not a diff.
MAX_DIRTY_FILES = 20

_GIT_TIMEOUT_S = 10


def _git(*args: str, repo: Optional[Path] = None, strip: bool = True) -> Optional[str]:
    """Run a read-only git command, or return None if it cannot be answered.

    Provenance collection must never be the reason a scoring run fails, so
    every failure mode here — no git, no repo, a timeout — is an absent field.

    `strip=False` is required for `status --porcelain`, whose status field is
    two columns wide and *left*-padded: an unstaged modification is
    ``" M path"``. Stripping that leading space shifts every subsequent column
    and silently eats the first character of the path — it turned
    ``eval/benchmark.py`` into ``val/benchmark.py``.
    """
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=str(repo or Path(__file__).resolve().parent.parent),
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_S,
        )
    except Exception:
        return None
    if completed.returncode != 0:
        return None
    out = completed.stdout.strip() if strip else completed.stdout.rstrip("\n")
    return out or None


def git_provenance(repo: Optional[Path] = None) -> Dict[str, object]:
    """Commit, branch and working-tree state of the harness that produced a run."""
    commit = _git("rev-parse", "HEAD", repo=repo)
    if commit is None:
        return {"available": False}

    status = _git("status", "--porcelain", repo=repo, strip=False)
    dirty_files: List[str] = [
        line[3:] for line in (status or "").splitlines() if line.strip()
    ]

    return {
        "available": True,
        "commit": commit,
        "commit_short": commit[:8],
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD", repo=repo),
        "describe": _git("describe", "--always", "--dirty", repo=repo),
        "committed_at": _git("log", "-1", "--format=%cI", repo=repo),
        # A number measured on a dirty tree cannot be tied to any commit. Say so
        # rather than reporting the commit alone and implying it can.
        "dirty": bool(dirty_files),
        "dirty_files": sorted(dirty_files)[:MAX_DIRTY_FILES],
        "dirty_file_count": len(dirty_files),
    }


def _primary_ipv4() -> Optional[str]:
    """The address this host presents on the LAN, without requiring the LAN.

    Uses a UDP socket, which picks a route without sending anything; falls back
    to a hostname lookup, then to nothing. Loopback is reported as absent — it
    identifies no machine, and AGENTS.md wants an address a reader can match to
    a workstation.
    """
    for resolve in (
        lambda: _route_lookup(),
        lambda: socket.gethostbyname(socket.gethostname()),
    ):
        try:
            address = resolve()
        except Exception:
            continue
        if address and not address.startswith("127."):
            return address
    return None


def _route_lookup() -> str:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("192.168.0.1", 1))
        return str(sock.getsockname()[0])
    finally:
        sock.close()


def machine_provenance() -> Dict[str, object]:
    """Which machine, which interpreter, which stack.

    `python_executable` is load-bearing rather than trivia: `.venv` here is
    `--system-site-packages`, so the path is what distinguishes a venv run from
    a bare global-interpreter one, and the two do not resolve the same packages
    (MIGRATION.md, Environment pinning).
    """
    from eval.runtime_guard import probe_graph_ops

    return {
        "hostname": socket.gethostname(),
        "ipv4": _primary_ipv4(),
        "user": _safe(getpass.getuser),
        "platform": platform.platform(),
        "processor": platform.processor() or None,
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "graph_ops": probe_graph_ops(),
    }


def _safe(fn):
    try:
        return fn()
    except Exception:
        return None


def run_provenance(repo: Optional[Path] = None) -> Dict[str, object]:
    """Everything needed to answer 'where did this number come from'.

    Goes into the scorecard next to the results. Collection never raises: an
    unanswerable field is recorded as absent, because losing a completed
    scoring run to a provenance lookup would be a poor trade.
    """
    return {
        "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": _safe(lambda: git_provenance(repo)) or {"available": False},
        "machine": _safe(machine_provenance) or {},
        "command": {
            "argv": list(sys.argv),
            "cwd": os.getcwd(),
        },
    }
