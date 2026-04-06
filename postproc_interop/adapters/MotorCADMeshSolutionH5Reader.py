from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from postproc_interop.adapters.MeshReader import MeshReader
from postproc_interop.model.MeshMat import MeshMat
from postproc_interop.model.MeshSolution import MeshSolution
from postproc_interop.model.SolutionMat import SolutionMat

# (label, unit) per H5 field key
_FIELD_META: dict[str, tuple[str, str]] = {
    "bx": ("Bx", "T"),
    "by": ("By", "T"),
    "a":  ("A",  "Wb/m"),
    "j":  ("J",  "A/mm2"),
}


class MotorCADMeshSolutionH5Reader(MeshReader):
    """Read MotorCAD-like HDF5 postprocess output into MeshSolution.

    H5 layout expected:
        mesh/tri_index, mesh/node_1..3, mesh/reg_code
        mesh/node_id, mesh/node_x_mm, mesh/node_y_mm
        regions/reg_code, regions/name  (optional)
        fields/bx, fields/by, fields/a, fields/j  (optional)
    """

    def can_read(self, path: Path) -> bool:
        return path.suffix.lower() in {".h5", ".hdf5"}

    def read(self, path: Path) -> MeshSolution:
        path = Path(path)
        with h5py.File(path, "r") as f:
            mesh_attrs = {"source_path": str(path)}
            if "mesh/moving_reg_codes" in f:
                mesh_attrs["moving_reg_codes"] = np.asarray(
                    f["mesh/moving_reg_codes"][:],
                    dtype=np.int32,
                )
            if "regions/reg_code" in f and "regions/name" in f:
                region_codes = np.asarray(f["regions/reg_code"][:], dtype=np.int32)
                region_names = np.asarray(f["regions/name"][:])
                mesh_attrs["region_name_by_code"] = {
                    int(code): (
                        name.decode("utf-8", errors="replace")
                        if isinstance(name, bytes) else str(name)
                    )
                    for code, name in zip(region_codes.tolist(), region_names.tolist())
                }

            mesh = MeshMat(
                tri_index=np.asarray(f["mesh/tri_index"][:]),
                node_1=np.asarray(f["mesh/node_1"][:]),
                node_2=np.asarray(f["mesh/node_2"][:]),
                node_3=np.asarray(f["mesh/node_3"][:]),
                reg_code=np.asarray(f["mesh/reg_code"][:]),
                node_id=np.asarray(f["mesh/node_id"][:]),
                x_mm=np.asarray(f["mesh/node_x_mm"][:]),
                y_mm=np.asarray(f["mesh/node_y_mm"][:]),
                region_code=(
                    np.asarray(f["regions/reg_code"][:])
                    if "regions/reg_code" in f else None
                ),
                region_name=(
                    np.asarray(f["regions/name"][:])
                    if "regions/name" in f else None
                ),
                attrs=mesh_attrs,
            )

            solution_dict: dict[str, SolutionMat] = {}
            for key, (label, unit) in _FIELD_META.items():
                h5_key = f"fields/{key}"
                if h5_key in f:
                    solution_dict[key] = SolutionMat(
                        label=label,
                        field=np.asarray(f[h5_key][:]),
                        unit=unit,
                    )

            attrs: dict = {"source_path": str(path)}
            if "moving_reg_codes" in mesh_attrs:
                attrs["moving_reg_codes"] = np.asarray(
                    mesh_attrs["moving_reg_codes"],
                    dtype=np.int32,
                )
            if "region_name_by_code" in mesh_attrs:
                attrs["region_name_by_code"] = dict(
                    mesh_attrs["region_name_by_code"]
                )
            if "steps" in f:
                attrs["steps"] = np.asarray(f["steps"][:])

        return MeshSolution(mesh=mesh, solution_dict=solution_dict, attrs=attrs)
