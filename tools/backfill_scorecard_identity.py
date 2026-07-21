"""Backfill checkpoint identity into scorecards written before identity existed.

Scorecards recorded only a checkpoint *path*. Where the path still resolves to
exactly one registered checkpoint, the identity can be recovered without
rerunning anything. Where the filename is one of the ambiguous ones, the
headline metric decides -- and the scorecard records *which* of the two methods
was used, so a reader can tell a measured fact from an inference.

Nothing is guessed silently: an entry that cannot be resolved is written as
``"sha256": null`` with the reason, not omitted.

    python -m tools.backfill_scorecard_identity            # dry run
    python -m tools.backfill_scorecard_identity --write
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.ckpt_identity import load_registry  # noqa: E402

TOL = 0.01  # percentage points; the registry stores metrics rounded to 3 decimals


def candidates(filename: str, registry: Dict[str, object]) -> List[Dict[str, object]]:
    return [
        e for e in registry.get("checkpoints", [])
        if filename == e.get("canonical_name") or filename in e.get("seen_as", [])
    ]


def headline(card: Dict[str, object]) -> Optional[float]:
    try:
        return float(card["summary"]["overall"]["Bnorm"]["nrmse_pct"])
    except (KeyError, TypeError, ValueError):
        return None


def resolve(card: Dict[str, object], registry: Dict[str, object]) -> Dict[str, object]:
    ckpt = (card.get("model") or {}).get("checkpoint")
    if not ckpt:
        return {}
    name = Path(str(ckpt).replace("\\", "/")).name
    cands = candidates(name, registry)

    if len(cands) == 1:
        entry, how = cands[0], "unique filename in registry"
    elif len(cands) > 1:
        got = headline(card)
        matched = [
            e for e in cands
            if e.get("metrics") and got is not None
            and abs(float(e["metrics"]["B_nrmse_pct"]) - got) <= TOL
        ]
        if len(matched) != 1:
            return {
                "sha256": None,
                "backfilled": True,
                "unresolved_reason": (
                    f"filename {name!r} is ambiguous ({len(cands)} checkpoints) and the "
                    f"headline |B| nRMSE {got} matched {len(matched)} of them"
                ),
            }
        entry, how = matched[0], f"ambiguous filename disambiguated by |B| nRMSE {got:.3f}%"
    else:
        return {
            "sha256": None,
            "backfilled": True,
            "unresolved_reason": f"no registered checkpoint is known by the name {name!r}",
        }

    return {
        "sha256": entry["sha256"],
        "bytes": entry.get("bytes"),
        "epoch": entry.get("epoch"),
        "filename": name,
        "canonical_name": entry.get("canonical_name"),
        "machine": entry.get("machine"),
        "registered": True,
        "backfilled": True,
        "resolved_by": how,
    }


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--glob", default="results/benchmark_v2_*.json")
    ap.add_argument("--write", action="store_true", help="apply; otherwise dry run")
    args = ap.parse_args(argv)

    registry = load_registry()
    touched = 0
    for path in sorted(Path().glob(args.glob)):
        doc = json.loads(path.read_text(encoding="utf-8"))
        changed = False
        for name, card in (doc.get("models") or {}).items():
            if not isinstance(card, dict) or "identity" in (card.get("model") or {}):
                continue
            ident = resolve(card, registry)
            if not ident:
                continue
            card["model"]["identity"] = ident
            changed = True
            mark = ident.get("sha256")
            print(f"  {path.name:42s} {name:26s} "
                  f"{(mark[:12] + '...') if mark else 'UNRESOLVED'}  {ident.get('resolved_by') or ident.get('unresolved_reason')}")
        if changed:
            touched += 1
            if args.write:
                path.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"\n{touched} scorecard(s) {'updated' if args.write else 'would change (dry run)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
