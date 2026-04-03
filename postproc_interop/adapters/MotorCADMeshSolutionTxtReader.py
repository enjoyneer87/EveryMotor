from __future__ import annotations

import io
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from postproc_interop.adapters.MeshReader import MeshReader
from postproc_interop.model.MeshMat import MeshMat
from postproc_interop.model.MeshSolution import MeshSolution
from postproc_interop.model.SolutionMat import SolutionMat

_ELEM_KEYS = frozenset({"TriIndex", "Node1", "Node2", "Node3", "RegCode", "Bx", "By", "A", "J"})
_NODE_KEYS = frozenset({"NodeIndex", "X", "Y"})
_REGION_KEYS = frozenset({"RegionCode", "RegionName"})

# Fallback positional indices when column name is absent from header
# (matches MotorCAD "new format" 9-column ElementsTable order)
_ELEM_FALLBACK  = {"TriIndex": 0, "Node1": 1, "Node2": 2, "Node3": 3,
                   "RegCode": 4, "Bx": 5, "By": 6, "A": 7, "J": 8}
_NODE_FALLBACK  = {"NodeIndex": 0, "X": 1, "Y": 2}
_REGION_FALLBACK = {"RegionCode": 0}

_FIELD_META: dict[str, tuple[str, str]] = {
    "Bx": ("Bx", "T"),
    "By": ("By", "T"),
    "A":  ("A",  "Wb/m"),
    "J":  ("J",  "A/mm2"),
}


@contextmanager
def _open_mcad_text(path: Path):
    """Open MotorCAD TXT robustly across Windows locales (UTF-16 BOM or cp949)."""
    with open(path, "rb") as fb:
        head = fb.read(4)
        fb.seek(0)
        if head.startswith((b"\xff\xfe", b"\xfe\xff")):
            encoding = "utf-16"
        elif head.startswith(b"\xef\xbb\xbf"):
            encoding = "utf-8-sig"
        elif b"\x00" in head:
            encoding = "utf-16"
        else:
            encoding = "cp949"
        wrapper = io.TextIOWrapper(fb, encoding=encoding, errors="replace", newline="")
        try:
            yield wrapper
        finally:
            try:
                wrapper.detach()
            except Exception:
                pass


def _is_table_header(line: str, table_name: str) -> bool:
    tokens = line.strip().split()
    return len(tokens) >= 3 and tokens[1].isdigit() and tokens[2].strip() == table_name


def _scan_to_table(in_file, table_name: str) -> str | None:
    while True:
        line = in_file.readline()
        if not line:
            return None
        if _is_table_header(line, table_name):
            return line


def _read_col_indices(in_file, expected_keys: frozenset) -> dict[str, int]:
    """Read the 4-line preamble after a table section header.

    MotorCAD TXT preamble structure:
        line 1: blank
        line 2: column names  ← parsed here (eMach skips this line)
        line 3: units
        line 4: separator (----)
    Returns {column_name: column_index} for names in expected_keys.
    """
    in_file.readline()            # blank
    col_line = in_file.readline() # column names
    in_file.readline()            # units
    in_file.readline()            # separator
    tokens = [t.strip() for t in col_line.split(",")]
    return {t: i for i, t in enumerate(tokens) if t in expected_keys}


class MotorCADMeshSolutionTxtReader(MeshReader):
    """Read first step block of MotorCAD FEA TXT export into MeshSolution.

    Column positions are resolved dynamically from the header name line
    (which eMach currently skips). Falls back to positional indices when
    a column name is absent.
    """

    def can_read(self, path: Path) -> bool:
        return path.suffix.lower() == ".txt"

    def read(self, path: Path) -> MeshSolution:
        path = Path(path)

        tri_indices: list[int] = []
        node_1s: list[int] = []
        node_2s: list[int] = []
        node_3s: list[int] = []
        reg_codes: list[int] = []
        field_vals: dict[str, list[float]] = {k: [] for k in _FIELD_META}

        node_ids: list[int] = []
        x_mms: list[float] = []
        y_mms: list[float] = []

        region_codes: list[int] = []
        region_names: list[str] = []

        with _open_mcad_text(path) as f:
            # ── ElementsTable ────────────────────────────────────
            hdr = _scan_to_table(f, "ElementsTable")
            if hdr is None:
                raise ValueError(f"ElementsTable not found in {path}")
            n_elem = int(hdr.strip().split()[1])

            ci = _read_col_indices(f, _ELEM_KEYS)
            ti_i = ci.get("TriIndex", _ELEM_FALLBACK["TriIndex"])
            n1_i = ci.get("Node1",    _ELEM_FALLBACK["Node1"])
            n2_i = ci.get("Node2",    _ELEM_FALLBACK["Node2"])
            n3_i = ci.get("Node3",    _ELEM_FALLBACK["Node3"])
            rc_i = ci.get("RegCode",  _ELEM_FALLBACK["RegCode"])
            field_idx = {k: ci.get(k, _ELEM_FALLBACK[k]) for k in _FIELD_META}

            for _ in range(n_elem):
                row = f.readline().split(",")
                if len(row) <= rc_i:
                    continue
                try:
                    tri_indices.append(int(row[ti_i]))
                    node_1s.append(int(row[n1_i]))
                    node_2s.append(int(row[n2_i]))
                    node_3s.append(int(row[n3_i]))
                    reg_codes.append(int(row[rc_i]))
                except (ValueError, IndexError):
                    continue
                for k, idx in field_idx.items():
                    try:
                        field_vals[k].append(float(row[idx]))
                    except (ValueError, IndexError):
                        pass

            # ── NodesTable ───────────────────────────────────────
            hdr = _scan_to_table(f, "NodesTable")
            if hdr is not None:
                n_nodes = int(hdr.strip().split()[1])
                ni_map = _read_col_indices(f, _NODE_KEYS)
                ni_i = ni_map.get("NodeIndex", _NODE_FALLBACK["NodeIndex"])
                x_i  = ni_map.get("X",         _NODE_FALLBACK["X"])
                y_i  = ni_map.get("Y",          _NODE_FALLBACK["Y"])
                for _ in range(n_nodes):
                    row = f.readline().split(",")
                    try:
                        node_ids.append(int(row[ni_i]))
                        x_mms.append(float(row[x_i]))
                        y_mms.append(float(row[y_i]))
                    except (ValueError, IndexError):
                        pass

            # ── RegionsTable ─────────────────────────────────────
            hdr = _scan_to_table(f, "RegionsTable")
            if hdr is not None:
                n_regions = int(hdr.strip().split()[1])
                rg_map = _read_col_indices(f, _REGION_KEYS)
                rc2_i = rg_map.get("RegionCode", _REGION_FALLBACK["RegionCode"])
                rn_i  = rg_map.get("RegionName")
                for _ in range(n_regions):
                    row = f.readline().split(",")
                    try:
                        region_codes.append(int(row[rc2_i]))
                    except (ValueError, IndexError):
                        continue
                    name = (row[rn_i].strip() if rn_i is not None and rn_i < len(row)
                            else row[-1].strip() if row else "")
                    region_names.append(name)

        # ── Assemble MeshMat ──────────────────────────────────────
        mesh = MeshMat(
            node_id=np.array(node_ids,    dtype=np.int32),
            x_mm=np.array(x_mms,         dtype=np.float64),
            y_mm=np.array(y_mms,         dtype=np.float64),
            tri_index=np.array(tri_indices, dtype=np.int32),
            node_1=np.array(node_1s,     dtype=np.int32),
            node_2=np.array(node_2s,     dtype=np.int32),
            node_3=np.array(node_3s,     dtype=np.int32),
            reg_code=np.array(reg_codes, dtype=np.int32),
            region_code=np.array(region_codes, dtype=np.int32) if region_codes else None,
            region_name=np.array(region_names, dtype=object)   if region_names else None,
            attrs={"source_path": str(path)},
        )

        # ── Assemble solution_dict ────────────────────────────────
        n = len(tri_indices)
        solution_dict: dict[str, SolutionMat] = {}
        for k, (label, unit) in _FIELD_META.items():
            vals = field_vals[k]
            if vals and len(vals) == n:
                solution_dict[k.lower()] = SolutionMat(
                    label=label,
                    field=np.array(vals, dtype=np.float64),
                    unit=unit,
                )

        return MeshSolution(
            mesh=mesh,
            solution_dict=solution_dict,
            attrs={"source_path": str(path)},
        )
