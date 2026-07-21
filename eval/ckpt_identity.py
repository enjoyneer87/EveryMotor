"""Checkpoint identity: content hash, registry lookup, and mismatch refusal.

Why this exists
---------------
Checkpoint filenames are not unique across machines. `mgn_nodeB_notime.pt`
already meant two different things at the same time: epoch 34 (interrupted, on
the department share) and epoch 55 (completed, on `HPC_134`). Scorecards
recorded only the checkpoint *path*, so a scorecard could not be traced back to
the weights that produced it, and a machine that pulled the share got different
weights than the one that wrote the numbers -- silently, with no error.

Two things fix that, and both live here:

1. Every scorecard records the checkpoint's sha256 and epoch, so the artifact
   identifies its own weights.
2. `results/checkpoints.json` is a registry of known checkpoints, tracked in
   git. The weights themselves are gitignored (28 MB each) and travel by the
   share, but their *identity* travels by `git pull`. A hash that is not in the
   registry is unknown, not wrong -- but a hash that contradicts the registry is
   an error worth stopping for.

This module deliberately does not import torch. `eval/` is verified to stay
torch-free (`tests/test_eval_feature_guard.py`), so identity can be checked on a
host that cannot even load the checkpoint.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional

REGISTRY_PATH = Path(__file__).resolve().parent.parent / "results" / "checkpoints.json"

#: Filename pattern the project uses. The epoch and the machine are part of the
#: name so that two runs cannot collide on one filename in the first place.
NAMING_CONVENTION = "<arch>_<variant>_ep<NN>_<machine>.pt"

_CHUNK = 1 << 20


class CheckpointIdentityError(ValueError):
    """Raised when a checkpoint's content contradicts the registry."""


def sha256_file(path: Path) -> str:
    """Content hash of a checkpoint. Streamed -- these files are ~28 MB."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def load_registry(path: Optional[Path] = None) -> Dict[str, object]:
    """Read the checkpoint registry. A missing registry is empty, not an error --
    a fresh clone that has not synced the share yet still has to run."""
    p = Path(path) if path is not None else REGISTRY_PATH
    if not p.exists():
        return {"version": "checkpoints/v1", "checkpoints": [], "ambiguous_names": {}}
    return json.loads(p.read_text(encoding="utf-8"))


def find_by_sha(sha256: str, registry: Optional[Dict[str, object]] = None) -> Optional[Dict[str, object]]:
    reg = registry if registry is not None else load_registry()
    for entry in reg.get("checkpoints", []):
        if entry.get("sha256") == sha256:
            return entry
    return None


def ambiguous_names(registry: Optional[Dict[str, object]] = None) -> Dict[str, object]:
    """Filenames that are known to have meant more than one thing.

    Seeing one of these is not fatal by itself -- the hash still decides -- but
    it means the name carries no information and should be renamed.
    """
    reg = registry if registry is not None else load_registry()
    return dict(reg.get("ambiguous_names", {}))


def identify(
    path: Path,
    epoch: Optional[int] = None,
    registry: Optional[Dict[str, object]] = None,
    strict: bool = True,
) -> Dict[str, object]:
    """Identify a checkpoint file by content.

    Returns the identity block that goes into a scorecard: sha256, size, the
    epoch as reported by the checkpoint itself, and what the registry knows.

    With ``strict`` (the default), a hash that the registry knows under a
    *different* epoch raises `CheckpointIdentityError` -- that combination means
    either the registry or the file is wrong, and scoring it would produce a
    number nobody can trace. An unknown hash is allowed: new runs are normal.
    """
    p = Path(path)
    sha = sha256_file(p)
    reg = registry if registry is not None else load_registry()
    entry = find_by_sha(sha, reg)

    identity: Dict[str, object] = {
        "sha256": sha,
        "bytes": p.stat().st_size,
        "epoch": epoch,
        "filename": p.name,
        "registered": entry is not None,
    }

    if entry is not None:
        identity["canonical_name"] = entry.get("canonical_name")
        identity["machine"] = entry.get("machine")
        if epoch is not None and entry.get("epoch") is not None and int(entry["epoch"]) != int(epoch):
            msg = (
                f"Checkpoint {p.name} hashes to {sha[:12]}..., which the registry "
                f"records as epoch {entry['epoch']}, but the file reports epoch {epoch}. "
                "One of the two is wrong; refusing to score an untraceable checkpoint."
            )
            if strict:
                raise CheckpointIdentityError(msg)
            identity["warning"] = msg

    amb = reg.get("ambiguous_names", {})
    if p.name in amb:
        identity["name_is_ambiguous"] = True
        identity["ambiguity_note"] = amb[p.name].get("reason")

    return identity


def format_identity(identity: Dict[str, object]) -> str:
    """One-line human summary for console output."""
    sha = str(identity.get("sha256", ""))[:12]
    ep = identity.get("epoch")
    bits: List[str] = [f"sha256={sha}...", f"epoch={ep}"]
    if identity.get("registered"):
        bits.append(f"registered as {identity.get('canonical_name')}")
    else:
        bits.append("not in registry")
    if identity.get("name_is_ambiguous"):
        bits.append("NAME IS AMBIGUOUS -- rename before sharing")
    return "  ".join(bits)
