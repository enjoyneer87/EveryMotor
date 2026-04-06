from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SolverMetadata:
    """Canonical solver/file semantics owned by postproc_interop."""

    source_file_name: str
    fidelity_type: str
    step_index: int
    step_semantics: str
    coupling_policy: str
    source_path: str = ""
    reader_name: str = ""
