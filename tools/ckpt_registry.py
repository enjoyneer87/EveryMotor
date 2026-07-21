"""Build and update `results/checkpoints.json`, the checkpoint identity registry.

The weights are gitignored (28 MB each) and travel by the department share.
Their identity travels by git. This tool is what keeps the two in step.

    # register everything under results/, reading each checkpoint's own epoch
    python -m tools.ckpt_registry --scan results

    # additionally hash a directory we cannot torch.load cheaply (the share).
    # Files there are resolved by sha256 against what is already known, so a
    # copy that matches a known checkpoint inherits its identity for free.
    python -m tools.ckpt_registry --scan results --hash-only "//192.168.0.165/.../checkpoints"

Hand-written fields (`scorecard`, `metrics`, `note`, `machine`) are preserved
across runs -- the tool only ever fills in what it can measure.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.ckpt_identity import NAMING_CONVENTION, sha256_file  # noqa: E402

MEASURED = ("sha256", "bytes", "epoch", "seen_as")
NAME_RE = re.compile(r"^(?P<stem>.+?)_ep(?P<epoch>\d+)_(?P<machine>[A-Za-z0-9]+)\.pt$")


def read_epoch(path: Path) -> Optional[int]:
    """The epoch the checkpoint reports about itself. Needs torch, so this is a
    tool concern, not an `eval/` concern."""
    try:
        import torch
    except ImportError:
        return None
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:  # a file we cannot read is a fact, not a crash
        print(f"  [WARN] {path.name}: cannot load ({type(exc).__name__})")
        return None
    ep = ckpt.get("epoch")
    return int(ep) if ep is not None else None


def canonical_name(stem_name: str, epoch: Optional[int], machine: str) -> str:
    """Apply the naming convention to a legacy filename."""
    m = NAME_RE.match(stem_name)
    if m:
        return stem_name
    base = stem_name[:-3] if stem_name.endswith(".pt") else stem_name
    base = re.sub(r"_epoch\d+.*$", "", base)
    ep = f"_ep{epoch}" if epoch is not None else ""
    return f"{base}{ep}_{machine}.pt"


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scan", type=Path, nargs="+", default=[Path("results")],
                    help="directories to scan and torch.load for the epoch")
    ap.add_argument("--hash-only", type=Path, nargs="*", default=[],
                    help="directories to hash without loading (e.g. the network share)")
    ap.add_argument("--machine", default="HPC_134", help="machine tag for newly discovered checkpoints")
    ap.add_argument("--out", type=Path, default=Path("results/checkpoints.json"))
    args = ap.parse_args(argv)

    out = Path(args.out)
    reg: Dict[str, object] = (
        json.loads(out.read_text(encoding="utf-8")) if out.exists()
        else {"version": "checkpoints/v1", "naming": NAMING_CONVENTION,
              "checkpoints": [], "ambiguous_names": {}}
    )
    by_sha = {e["sha256"]: e for e in reg.get("checkpoints", [])}

    for d in args.scan:
        for p in sorted(Path(d).glob("*.pt")):
            sha = sha256_file(p)
            entry = by_sha.setdefault(sha, {"sha256": sha})
            entry["bytes"] = p.stat().st_size
            if entry.get("epoch") is None:
                entry["epoch"] = read_epoch(p)
            entry.setdefault("machine", args.machine)
            entry.setdefault("canonical_name",
                             canonical_name(p.name, entry.get("epoch"), entry["machine"]))
            seen = set(entry.get("seen_as", [])) | {p.name}
            entry["seen_as"] = sorted(seen)
            print(f"  {p.name:46s} sha={sha[:12]} epoch={entry.get('epoch')}")

    for d in args.hash_only:
        d = Path(d)
        if not d.exists():
            print(f"  [SKIP] unreachable: {d}")
            continue
        for p in sorted(d.glob("*.pt")):
            sha = sha256_file(p)
            entry = by_sha.get(sha)
            if entry is None:
                entry = by_sha.setdefault(sha, {"sha256": sha, "bytes": p.stat().st_size,
                                                "epoch": None, "machine": "unknown"})
                entry.setdefault("canonical_name", p.name)
                print(f"  {p.name:46s} sha={sha[:12]} UNKNOWN -- not seen locally")
            else:
                print(f"  {p.name:46s} sha={sha[:12]} -> {entry['canonical_name']}")
            entry["seen_as"] = sorted(set(entry.get("seen_as", [])) | {p.name})

    # A filename that maps to more than one hash is ambiguous. Record it so the
    # loader can say so instead of silently scoring the wrong weights.
    amb: Dict[str, object] = dict(reg.get("ambiguous_names", {}))
    name_to_shas: Dict[str, List[str]] = {}
    for e in by_sha.values():
        for nm in e.get("seen_as", []):
            name_to_shas.setdefault(nm, []).append(e["sha256"])
    for nm, shas in name_to_shas.items():
        if len(set(shas)) > 1:
            amb[nm] = {
                "reason": f"{len(set(shas))} different checkpoints have been stored under this name",
                "seen_as": sorted(
                    f"{s[:12]}... (epoch {by_sha[s].get('epoch')}, {by_sha[s].get('machine')})"
                    for s in set(shas)
                ),
            }
    reg["ambiguous_names"] = amb
    reg["naming"] = NAMING_CONVENTION
    reg["checkpoints"] = sorted(by_sha.values(), key=lambda e: (e.get("canonical_name") or ""))

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(reg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\n{out}: {len(reg['checkpoints'])} checkpoints, "
          f"{len(reg['ambiguous_names'])} ambiguous name(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
