"""Motor-CAD DOE H5 loading for the benchmark harness.

Two H5 layouts, one reader
--------------------------
The DOE export writes two different formats, and the training loaders only
understood the first:

``pyMCAD.magnetic.timeseries.moving_mesh`` (``Mag_OnLoadTorque_*``)
    ``steps`` dataset, ``fields/*`` shaped ``(n_steps, n_elements)``, plus
    per-step moving-node coordinates.

``pyMCAD.magnetic.static_mesh`` (``Mag_StaticLoad_*``, ``Mag_StaticOC_*``)
    scalar ``step`` dataset, ``fields/*`` shaped ``(n_elements,)``, no motion.

`parse_h5_timeseries` in `train_doe_meshgraphnet.py` raises
``Invalid H5 (missing 'steps')`` on the second layout, and the caller catches the
exception and prints a warning — which is why roughly two thirds of the exported
solves were being dropped silently. This reader accepts both and records which
layout each sample came from.

Fields are stored **per element**, matching the Motor-CAD export. Nothing here
scatters to nodes; that is the graph builder's job, and keeping the element view
intact is what lets the torque operator integrate over the airgap band.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import h5py
import numpy as np

from eval.mesh_regions import RegionGrouping, build_region_grouping, element_centroids_m, read_region_names

FIELD_KEYS: Tuple[str, ...] = ("bx", "by", "a", "j")

TIMESERIES_FORMAT = "timeseries"
STATIC_FORMAT = "static"


class DoeDataError(ValueError):
    """Raised when a DOE H5 file cannot be interpreted."""


@dataclass(frozen=True)
class CaseMesh:
    """Immutable mesh topology shared by every timestep of one solve.

    Node arrays are ordered by ascending ``mesh/node_id`` and element
    connectivity is expressed in that ordering, matching the convention the
    training graph builders use.
    """

    node_x_mm: np.ndarray
    node_y_mm: np.ndarray
    tri: Tuple[np.ndarray, np.ndarray, np.ndarray]
    reg_code: np.ndarray
    name_of_code: Mapping[int, str]
    moving_reg_codes: Tuple[int, ...]
    moving_node_indices: Optional[np.ndarray]

    @property
    def n_nodes(self) -> int:
        return int(self.node_x_mm.size)

    @property
    def n_elements(self) -> int:
        return int(self.reg_code.size)

    def region_grouping(self) -> RegionGrouping:
        cx, cy = element_centroids_m(self.node_x_mm, self.node_y_mm, self.tri)
        return build_region_grouping(self.reg_code, self.name_of_code, np.hypot(cx, cy))


@dataclass(frozen=True)
class CaseSample:
    """One solved field state: a mesh position plus its element-wise fields."""

    case_index: int
    source_type: str
    step_index: int
    layout: str
    time_s: float
    rotate_step: float
    node_x_mm: np.ndarray
    node_y_mm: np.ndarray
    fields: Mapping[str, np.ndarray]

    def field_matrix(self, keys: Sequence[str]) -> np.ndarray:
        """Stack selected element fields into ``(n_elements, len(keys))``."""
        return np.stack([np.asarray(self.fields[k], dtype=np.float64) for k in keys], axis=1)


@dataclass(frozen=True)
class CaseRecord:
    """One solve file: shared mesh + every timestep in it."""

    case_index: int
    source_type: str
    path: Path
    mesh: CaseMesh
    samples: Tuple[CaseSample, ...]
    condition: Mapping[str, float]

    def __len__(self) -> int:
        return len(self.samples)


def classify_source_type(path_like: str) -> str:
    """Classify a Motor-CAD postprocess file by name (mirrors doe_data_utils)."""
    name = Path(str(path_like).replace("\\", "/")).name.lower()
    if "onloadtorque" in name:
        return "OnLoadTorque"
    if "losselement" in name or "onloadloss" in name:
        return "LossElement_OnLoadLoss"
    if "staticloadinductance" in name:
        return "StaticLoadInductance"
    if "staticload" in name:
        return "StaticLoad"
    if "staticoc" in name:
        return "StaticOC"
    return "Unknown"


def resolve_h5_path(recorded: str, data_dir: Path, case_index: int) -> Optional[Path]:
    """Resolve an H5 path recorded on another machine/OS.

    The manifest stores absolute Windows paths from the solve host; this tries
    the literal path first, then reconstructs it under ``data_dir``.
    """
    basename = str(recorded).replace("\\", "/").split("/")[-1]
    for candidate in (
        Path(recorded),
        data_dir / f"case_{case_index:04d}" / "postproc" / basename,
        data_dir / basename,
    ):
        if candidate.exists():
            return candidate
    return None


def _read_mesh(f: h5py.File) -> Tuple[CaseMesh, np.ndarray]:
    """Read topology, returning the mesh and the node_id sort order."""
    node_id = np.asarray(f["mesh/node_id"][:], dtype=np.int64)
    sort_order = np.argsort(node_id)
    sorted_ids = node_id[sort_order]

    # node_id -> position in the sorted ordering
    lut = np.full(int(sorted_ids.max()) + 1, -1, dtype=np.int64)
    lut[sorted_ids] = np.arange(sorted_ids.size, dtype=np.int64)

    n1 = np.asarray(f["mesh/node_1"][:], dtype=np.int64)
    n2 = np.asarray(f["mesh/node_2"][:], dtype=np.int64)
    n3 = np.asarray(f["mesh/node_3"][:], dtype=np.int64)
    reg = np.asarray(f["mesh/reg_code"][:], dtype=np.int64)

    m = min(n1.size, n2.size, n3.size, reg.size)
    hi = lut.size - 1
    i1 = lut[np.clip(n1[:m], 0, hi)]
    i2 = lut[np.clip(n2[:m], 0, hi)]
    i3 = lut[np.clip(n3[:m], 0, hi)]
    valid = (i1 >= 0) & (i2 >= 0) & (i3 >= 0)
    if not np.any(valid):
        raise DoeDataError("No valid elements after node_id remapping")

    moving_codes = (
        tuple(int(c) for c in f["mesh/moving_reg_codes"][:]) if "mesh/moving_reg_codes" in f else ()
    )
    moving_idx = (
        np.asarray(f["mesh/moving_node_indices"][:], dtype=np.int64)
        if "mesh/moving_node_indices" in f
        else None
    )

    mesh = CaseMesh(
        node_x_mm=np.asarray(f["mesh/node_x_mm"][:], dtype=np.float64)[sort_order],
        node_y_mm=np.asarray(f["mesh/node_y_mm"][:], dtype=np.float64)[sort_order],
        tri=(i1[valid], i2[valid], i3[valid]),
        reg_code=reg[:m][valid],
        name_of_code=read_region_names(f),
        moving_reg_codes=moving_codes,
        moving_node_indices=moving_idx,
    )
    return mesh, valid


def _read_field(f: h5py.File, key: str, n_elements: int) -> Optional[np.ndarray]:
    path = f"fields/{key}"
    if path not in f:
        return None
    arr = np.asarray(f[path][:], dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None, :]
    return arr


def read_case_file(
    path: Path,
    case_index: int,
    condition: Mapping[str, float],
    max_steps: Optional[int] = None,
) -> CaseRecord:
    """Read one Motor-CAD solve file into a `CaseRecord`.

    Handles both the moving-mesh timeseries and the static-mesh layouts.
    """
    path = Path(path)
    source_type = classify_source_type(path)

    with h5py.File(path, "r") as f:
        mesh, valid_elem = _read_mesh(f)
        n_elem_raw = int(valid_elem.size)

        if "steps" in f:
            layout = TIMESERIES_FORMAT
            steps = np.asarray(f["steps"][:], dtype=np.int64)
        elif "step" in f:
            layout = STATIC_FORMAT
            steps = np.asarray(f["step"][()], dtype=np.int64).reshape(1)
        else:
            raise DoeDataError(f"H5 has neither 'steps' nor 'step': {path}")

        raw_fields = {k: _read_field(f, k, n_elem_raw) for k in FIELD_KEYS}
        present = {k: v for k, v in raw_fields.items() if v is not None}
        if "bx" not in present or "by" not in present:
            raise DoeDataError(f"H5 missing fields/bx or fields/by: {path}")

        time_s = (
            np.asarray(f["meta/time_s"][:], dtype=np.float64) if "meta/time_s" in f else None
        )
        rotate = (
            np.asarray(f["meta/rotate_step"][:], dtype=np.float64)
            if "meta/rotate_step" in f
            else None
        )

        x_step = (
            np.asarray(f["mesh/node_x_mm_by_step_moving"][:], dtype=np.float64)
            if "mesh/node_x_mm_by_step_moving" in f
            else None
        )
        y_step = (
            np.asarray(f["mesh/node_y_mm_by_step_moving"][:], dtype=np.float64)
            if "mesh/node_y_mm_by_step_moving" in f
            else None
        )

        n_steps = int(steps.size)
        if max_steps is not None:
            n_steps = min(n_steps, int(max_steps))

        samples: List[CaseSample] = []
        for si in range(n_steps):
            node_x, node_y = _step_coordinates(mesh, x_step, y_step, si)
            fields = {
                k: v[si if v.shape[0] > 1 else 0][:n_elem_raw][valid_elem]
                for k, v in present.items()
            }
            samples.append(
                CaseSample(
                    case_index=int(case_index),
                    source_type=source_type,
                    step_index=int(steps[si] if si < steps.size else si),
                    layout=layout,
                    # time_s[0] is NaN in the export (no dt before the first step).
                    time_s=_finite_or(time_s[si] if time_s is not None and si < time_s.size else None),
                    rotate_step=_finite_or(
                        rotate[si] if rotate is not None and si < rotate.size else None
                    ),
                    node_x_mm=node_x,
                    node_y_mm=node_y,
                    fields=fields,
                )
            )

    return CaseRecord(
        case_index=int(case_index),
        source_type=source_type,
        path=path,
        mesh=mesh,
        samples=tuple(samples),
        condition=dict(condition),
    )


def _finite_or(value, default: float = 0.0) -> float:
    if value is None:
        return float(default)
    v = float(value)
    return v if np.isfinite(v) else float(default)


def _step_coordinates(
    mesh: CaseMesh,
    x_step: Optional[np.ndarray],
    y_step: Optional[np.ndarray],
    step: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Node coordinates at one step, applying moving-node displacement.

    ``mesh.node_*`` is stored in node_id-sorted order while
    ``moving_node_indices`` addresses the original file order, so the indices are
    translated before scattering.
    """
    x = mesh.node_x_mm.copy()
    y = mesh.node_y_mm.copy()
    idx = mesh.moving_node_indices
    if x_step is None or y_step is None or idx is None or step >= x_step.shape[0]:
        return x, y

    n = min(idx.size, x_step.shape[1], y_step.shape[1])
    target = idx[:n]
    xv, yv = x_step[step, :n], y_step[step, :n]
    ok = (target >= 0) & (target < x.size) & np.isfinite(xv) & np.isfinite(yv)
    x[target[ok]] = xv[ok]
    y[target[ok]] = yv[ok]
    return x, y


def iter_case_records(
    manifest: Mapping,
    data_dir: Path,
    case_indices: Optional[Sequence[int]] = None,
    source_types: Sequence[str] = ("OnLoadTorque",),
    max_steps: Optional[int] = None,
) -> Iterator[CaseRecord]:
    """Yield `CaseRecord`s for the requested cases and solve types.

    Unresolvable or unreadable files raise `DoeDataError` from the caller's
    perspective only if nothing else in the case worked; individual failures are
    reported through the returned record set being short, which the harness
    counts and writes into the results JSON. Silent skipping is what hid the
    static-mesh problem, so failures are surfaced via the `skipped` list on
    `load_doe_cases`.
    """
    for record, _ in _iter_with_failures(manifest, data_dir, case_indices, source_types, max_steps):
        yield record


def _iter_with_failures(
    manifest: Mapping,
    data_dir: Path,
    case_indices: Optional[Sequence[int]],
    source_types: Sequence[str],
    max_steps: Optional[int],
) -> Iterator[Tuple[CaseRecord, None]]:
    wanted = set(int(c) for c in case_indices) if case_indices is not None else None
    allowed = {str(s) for s in source_types}
    data_dir = Path(data_dir)

    for case in manifest.get("cases", []):
        index = int(case["index"])
        if wanted is not None and index not in wanted:
            continue
        condition = {**case.get("geometry", {}), **case.get("electrical", {})}
        for recorded in case.get("h5_paths") or []:
            if classify_source_type(recorded) not in allowed:
                continue
            resolved = resolve_h5_path(recorded, data_dir, index)
            if resolved is None:
                continue
            yield read_case_file(resolved, index, condition, max_steps=max_steps), None


@dataclass(frozen=True)
class DoeLoadReport:
    """What was loaded and, more importantly, what was not."""

    records: Tuple[CaseRecord, ...]
    skipped: Tuple[Tuple[int, str, str], ...]  # (case_index, path, reason)

    @property
    def n_samples(self) -> int:
        return sum(len(r) for r in self.records)

    def to_dict(self) -> Dict[str, object]:
        return {
            "n_records": len(self.records),
            "n_samples": self.n_samples,
            "n_skipped": len(self.skipped),
            "skipped": [
                {"case_index": c, "path": p, "reason": r} for c, p, r in self.skipped
            ],
        }


def load_doe_cases(
    manifest: Mapping,
    data_dir: Path,
    case_indices: Optional[Sequence[int]] = None,
    source_types: Sequence[str] = ("OnLoadTorque",),
    max_steps: Optional[int] = None,
) -> DoeLoadReport:
    """Load DOE cases, recording every skip with its reason.

    The skip list is written into the benchmark results so that a run over
    fewer samples than expected is visible in the artifact rather than only in
    stdout.
    """
    wanted = set(int(c) for c in case_indices) if case_indices is not None else None
    allowed = {str(s) for s in source_types}
    data_dir = Path(data_dir)

    records: List[CaseRecord] = []
    skipped: List[Tuple[int, str, str]] = []

    for case in manifest.get("cases", []):
        index = int(case["index"])
        if wanted is not None and index not in wanted:
            continue
        condition = {**case.get("geometry", {}), **case.get("electrical", {})}
        for recorded in case.get("h5_paths") or []:
            source = classify_source_type(recorded)
            if source not in allowed:
                continue
            resolved = resolve_h5_path(recorded, data_dir, index)
            if resolved is None:
                skipped.append((index, str(recorded), "file_not_found"))
                continue
            try:
                records.append(read_case_file(resolved, index, condition, max_steps=max_steps))
            except (DoeDataError, OSError, KeyError) as exc:
                skipped.append((index, str(resolved), f"{type(exc).__name__}: {exc}"))

    return DoeLoadReport(records=tuple(records), skipped=tuple(skipped))
