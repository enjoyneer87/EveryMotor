"""Preflight the graph-op stack the MeshGraphNet adapters depend on.

Why this exists
---------------
`physicsnemo` reaches PyTorch Geometric and `torch_scatter` through
``check_version_spec(..., hard_fail=False)``. When either is missing that call
returns ``False`` and nothing is raised: the model still constructs, the
checkpoint still loads, and only the *forward pass* raises — via
``physicsnemo.nn.module.gnn_layers.graph_types.raise_missing_pyg_error``.

`eval.benchmark.evaluate_predictor` catches per-sample exceptions on purpose, so
one broken sample cannot silently shrink the test set. That safeguard turns this
particular ImportError into a per-sample entry in `failures`; the model then
summarizes to ``n_samples: 0``, the run prints ``no samples``, and
`main` returns 0. A missing dependency is indistinguishable from a clean run
that happened to score nothing — which is how it got past a reviewer once
already (see AGENTS.md, Environment Rules).

This module moves that failure to checkpoint-load time, before any scoring, and
names the fix. `probe_graph_ops` is the non-raising half, so the scorecard can
record which stack actually produced the numbers.

Version pinning
---------------
`torch_scatter` ships compiled against one specific torch build — the wheel
installed here is ``2.1.2+pt211cu128``, i.e. torch 2.11 / CUDA 12.8. On
`HPC_134` torch is resolved from the *global* interpreter while `torch_scatter`
lives in `.venv` (`--system-site-packages`), so a global torch upgrade breaks
the pairing without touching the venv at all. `require_graph_ops` reports the
observed torch build alongside the failure for exactly that case.
"""

from __future__ import annotations

import importlib
from typing import Dict, Optional

# The packages `physicsnemo`'s MeshGraphNet path gates on. Names, not versions:
# the version spec is physicsnemo's business and is read from it below.
GRAPH_OP_PACKAGES = ("torch_geometric", "torch_scatter")


class GraphOpsUnavailable(RuntimeError):
    """Raised when the MeshGraphNet forward pass cannot work on this machine."""


def _version_of(name: str) -> Optional[str]:
    """Import a package and return its version, or None if it cannot be imported."""
    try:
        module = importlib.import_module(name)
    except Exception:
        return None
    return str(getattr(module, "__version__", "unknown"))


def _torch_build() -> Optional[str]:
    """The torch build string, for the ABI-mismatch message. None if torch is absent."""
    try:
        import torch
    except Exception:
        return None
    return str(torch.__version__)


def _physicsnemo_accepts_stack() -> Optional[bool]:
    """Whether physicsnemo itself considers the PyG stack usable.

    Reads the flag physicsnemo actually branches on rather than re-deriving the
    check here, so this guard cannot drift from the behavior it guards. Returns
    None when the flag cannot be read (older or newer physicsnemo layout), in
    which case importability is all we can assert.
    """
    try:
        from physicsnemo.nn.module.gnn_layers.graph_types import PYG_AVAILABLE
    except Exception:
        return None
    return bool(PYG_AVAILABLE)


def probe_graph_ops() -> Dict[str, object]:
    """Report the graph-op stack without raising.

    Returned dict is JSON-safe and goes into the scorecard: a number is only
    reproducible if the stack that produced it is on the record.
    """
    versions = {name: _version_of(name) for name in GRAPH_OP_PACKAGES}
    return {
        "packages": versions,
        "torch": _torch_build(),
        "physicsnemo": _version_of("physicsnemo"),
        "physicsnemo_accepts_stack": _physicsnemo_accepts_stack(),
        "usable": all(v is not None for v in versions.values())
        and _physicsnemo_accepts_stack() is not False,
    }


def require_graph_ops(context: str = "MeshGraphNet") -> Dict[str, object]:
    """Fail loudly if the MeshGraphNet forward pass cannot work here.

    Call this at checkpoint load, not at predict: the point is to fail before
    the harness starts folding per-sample ImportErrors into `failures`.
    """
    probe = probe_graph_ops()
    versions: Dict[str, Optional[str]] = probe["packages"]  # type: ignore[assignment]
    missing = [name for name, version in versions.items() if version is None]

    if missing:
        raise GraphOpsUnavailable(
            f"{context} cannot run: {', '.join(missing)} is not importable "
            f"(torch={probe['torch']}, physicsnemo={probe['physicsnemo']}).\n"
            "physicsnemo does not fail on this by itself -- it raises only inside the "
            "forward pass, which the benchmark catches per sample, so the run would "
            "report 'no samples' and exit 0 as if nothing were wrong.\n"
            "torch_scatter is built against one exact torch release; install the wheel "
            "matching the torch above (this repo is pinned to torch 2.11.0+cu128 / "
            "torch_scatter 2.1.2+pt211cu128 -- see MIGRATION.md, Environment pinning)."
        )

    if probe["physicsnemo_accepts_stack"] is False:
        installed = ", ".join(f"{name}={version}" for name, version in versions.items())
        raise GraphOpsUnavailable(
            f"{context} cannot run: physicsnemo {probe['physicsnemo']} rejects the "
            f"installed graph stack ({installed}, torch={probe['torch']}).\n"
            "Both packages import, so this is a version-spec mismatch rather than a "
            "missing install -- most often a torch upgrade that left torch_scatter "
            "compiled against the previous release. Reinstall torch_scatter for the "
            "torch build above (see MIGRATION.md, Environment pinning)."
        )

    return probe
