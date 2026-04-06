from __future__ import annotations

import importlib
import sys
from pathlib import Path

import h5py
import numpy as np

from postproc_interop.adapters.MeshReader import MeshReader
from postproc_interop.model.MeshMat import MeshMat
from postproc_interop.model.MeshSolution import MeshSolution
from postproc_interop.model.SolutionMat import SolutionMat


_FIELD_META: dict[str, tuple[str, str]] = {
    "bx": ("Bx", "T"),
    "by": ("By", "T"),
    "a": ("A", "Wb/m"),
    "j": ("J", "A/mm2"),
}


class MotorCADMeshSolutionH5PyMCADReader(MeshReader):
    """Read the first H5 step through eMach pyMCAD into MeshSolution."""

    def __init__(
        self,
        *,
        repo_root: Path | None = None,
        emach_root: Path | None = None,
    ) -> None:
        self._repo_root = Path(repo_root) if repo_root is not None else None
        self._emach_root = Path(emach_root) if emach_root is not None else None

    def can_read(self, path: Path) -> bool:
        return Path(path).suffix.lower() in {".h5", ".hdf5"}

    def read(self, path: Path) -> MeshSolution:
        pymcad = self._load_pymcad_module()
        raw_attrs = self._read_raw_h5_attrs(Path(path))
        ts = pymcad.get_magnetic_timeseries_from_file(str(Path(path)))
        steps = list(getattr(ts, "steps", []))
        if not steps:
            raise ValueError(f"No steps found in pyMCAD timeseries: {path}")

        step_key = int(steps[0])
        by_step = getattr(ts, "by_step", None)
        if by_step is not None:
            regions = by_step[step_key]
        else:
            regions = ts[step_key]
        node_xy = dict(getattr(regions, "node_xy", {}) or {})
        if not node_xy:
            raise ValueError(
                f"NodesTable coordinates are missing in pyMCAD data: {path}"
            )

        node_id = np.asarray(sorted(node_xy.keys()), dtype=np.int32)
        x_mm = np.asarray(
            [float(node_xy[int(nid)][0]) for nid in node_id.tolist()],
            dtype=np.float64,
        )
        y_mm = np.asarray(
            [float(node_xy[int(nid)][1]) for nid in node_id.tolist()],
            dtype=np.float64,
        )
        local_lut = {
            int(nid): int(idx)
            for idx, nid in enumerate(node_id.tolist())
        }

        tri_index: list[int] = []
        node_1: list[int] = []
        node_2: list[int] = []
        node_3: list[int] = []
        reg_code: list[int] = []
        bx: list[float] = []
        by: list[float] = []
        a: list[float] = []
        j: list[float] = []
        region_names: dict[int, str] = {}

        for region in getattr(regions, "_regions", []):
            if not getattr(region, "elements", None):
                continue
            region_code = int(
                getattr(region, "reg_code", 0)
                or region.elements[0].reg_code
            )
            region_name = str(getattr(region, "region_name", "") or "")
            if region_name:
                region_names[region_code] = region_name

            for element in region.elements:
                try:
                    local_n1 = local_lut[int(element.node_1)]
                    local_n2 = local_lut[int(element.node_2)]
                    local_n3 = local_lut[int(element.node_3)]
                except KeyError:
                    continue

                tri_index.append(int(element.tri_index))
                node_1.append(local_n1)
                node_2.append(local_n2)
                node_3.append(local_n3)
                reg_code.append(int(element.reg_code))
                bx.append(float(element.bx))
                by.append(float(element.by))
                a.append(float(element.a))
                j.append(float(element.j))

        if not tri_index:
            raise ValueError(
                f"No triangle elements were extracted from pyMCAD data: {path}"
            )

        order = np.argsort(
            np.asarray(tri_index, dtype=np.int64),
            kind="stable",
        )
        tri_index_arr = np.asarray(tri_index, dtype=np.int32)[order]
        node_1_arr = np.asarray(node_1, dtype=np.int32)[order]
        node_2_arr = np.asarray(node_2, dtype=np.int32)[order]
        node_3_arr = np.asarray(node_3, dtype=np.int32)[order]
        reg_code_arr = np.asarray(reg_code, dtype=np.int32)[order]

        mesh = MeshMat(
            tri_index=tri_index_arr,
            node_1=node_1_arr,
            node_2=node_2_arr,
            node_3=node_3_arr,
            reg_code=reg_code_arr,
            node_id=node_id,
            x_mm=x_mm,
            y_mm=y_mm,
            region_code=(
                np.asarray(sorted(region_names.keys()), dtype=np.int32)
                if region_names else None
            ),
            region_name=(
                np.asarray(
                    [region_names[key] for key in sorted(region_names.keys())],
                    dtype=object,
                )
                if region_names else None
            ),
            attrs={
                "source_path": str(Path(path)),
                "reader": "pyMCAD",
                "step": step_key,
                **raw_attrs,
            },
        )
        solution_dict = {
            "bx": SolutionMat(
                label=_FIELD_META["bx"][0],
                field=np.asarray(bx, dtype=np.float64)[order],
                unit=_FIELD_META["bx"][1],
            ),
            "by": SolutionMat(
                label=_FIELD_META["by"][0],
                field=np.asarray(by, dtype=np.float64)[order],
                unit=_FIELD_META["by"][1],
            ),
            "a": SolutionMat(
                label=_FIELD_META["a"][0],
                field=np.asarray(a, dtype=np.float64)[order],
                unit=_FIELD_META["a"][1],
            ),
            "j": SolutionMat(
                label=_FIELD_META["j"][0],
                field=np.asarray(j, dtype=np.float64)[order],
                unit=_FIELD_META["j"][1],
            ),
        }
        return MeshSolution(
            mesh=mesh,
            solution_dict=solution_dict,
            attrs={
                "source_path": str(Path(path)),
                "reader": "pyMCAD",
                "step": step_key,
                "steps": np.asarray([step_key], dtype=np.int32),
                **raw_attrs,
            },
        )

    def _read_raw_h5_attrs(self, path: Path) -> dict[str, object]:
        attrs: dict[str, object] = {}
        try:
            with h5py.File(path, "r") as handle:
                if "mesh/moving_reg_codes" in handle:
                    attrs["moving_reg_codes"] = np.asarray(
                        handle["mesh/moving_reg_codes"][:],
                        dtype=np.int32,
                    )
                if "regions/reg_code" in handle and "regions/name" in handle:
                    region_codes = np.asarray(
                        handle["regions/reg_code"][:],
                        dtype=np.int32,
                    )
                    region_names = handle["regions/name"][:]
                    attrs["region_name_by_code"] = {
                        int(code): (
                            name.decode("utf-8", errors="replace")
                            if isinstance(name, bytes) else str(name)
                        )
                        for code, name in zip(
                            region_codes.tolist(),
                            region_names.tolist(),
                        )
                    }
        except (OSError, KeyError, ValueError):
            return {}
        return attrs

    def _load_pymcad_module(self):
        emach_root = self._resolve_emach_root()
        marker = emach_root / "tools" / "motorCAD" / "pyMCAD" / "__init__.py"
        if not marker.exists():
            raise FileNotFoundError(
                f"pyMCAD package was not found under {emach_root}"
            )

        path_str = str(emach_root)
        inserted = False
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
            inserted = True
        try:
            return importlib.import_module("tools.motorCAD.pyMCAD")
        finally:
            if inserted:
                try:
                    sys.path.remove(path_str)
                except ValueError:
                    pass

    def _resolve_emach_root(self) -> Path:
        candidates: list[Path] = []
        if self._emach_root is not None:
            candidates.append(self._emach_root)
        if self._repo_root is not None:
            candidates.append(self._repo_root / "eMach")
        candidates.append(Path(__file__).resolve().parents[2] / "eMach")

        seen: set[str] = set()
        for candidate in candidates:
            resolved = (
                str(candidate.resolve())
                if candidate.exists()
                else str(candidate)
            )
            if resolved in seen:
                continue
            seen.add(resolved)
            if (candidate / "tools" / "motorCAD" / "pyMCAD").exists():
                return candidate

        raise FileNotFoundError(
            "Unable to resolve eMach root for pyMCAD import"
        )
