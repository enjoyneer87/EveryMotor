from __future__ import annotations

import argparse
from pathlib import Path

from .adapters.h5_motorcad import MotorCADH5Adapter
from .exporters.babylon_json import export_babylon_meshframe


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Postprocess interoperability CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_sum = sub.add_parser("summary", help="Read source and print summary")
    p_sum.add_argument("--input", required=True)

    p_bab = sub.add_parser(
        "export-babylon",
        help="Export source to Babylon JSON",
    )
    p_bab.add_argument("--input", required=True)
    p_bab.add_argument("--output", required=True)

    return p


def _read_any(path: Path):
    h5_adapter = MotorCADH5Adapter()
    if h5_adapter.can_read(path):
        return h5_adapter.read(path)
    raise ValueError(f"No adapter available for: {path}")


def main() -> None:
    args = build_parser().parse_args()
    src = Path(args.input)

    frame = _read_any(src)

    if args.cmd == "summary":
        print(frame.summary())
        return

    if args.cmd == "export-babylon":
        out = Path(args.output)
        export_babylon_meshframe(frame, out)
        print(f"Saved: {out}")
        return

    raise ValueError(f"Unhandled command: {args.cmd}")


if __name__ == "__main__":
    main()
