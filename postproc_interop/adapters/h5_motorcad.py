from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from postproc_interop.adapters.base import PostprocAdapter
from postproc_interop.model import MeshFrame, MeshTopology, NodeTable, RegionTable


class MotorCADH5Adapter(PostprocAdapter):
    """Read MotorCAD-like h5 postprocess outputs into MeshFrame."""

    def can_read(self, path: Path) -> bool:
        return path.suffix.lower() in {".h5", ".hdf5"}

    def read(self, path: Path) -> MeshFrame:
        with h5py.File(path, "r") as f:
            topology = MeshTopology(
                tri_index=np.asarray(f["mesh/tri_index"][:]),
                node_1=np.asarray(f["mesh/node_1"][:]),
                node_2=np.asarray(f["mesh/node_2"][:]),
                node_3=np.asarray(f["mesh/node_3"][:]),
                reg_code=np.asarray(f["mesh/reg_code"][:]),
            )
            nodes = NodeTable(
                node_id=np.asarray(f["mesh/node_id"][:]),
                x_mm=np.asarray(f["mesh/node_x_mm"][:]),
                y_mm=np.asarray(f["mesh/node_y_mm"][:]),
            )

            regions = None
            if "regions/reg_code" in f and "regions/name" in f:
                regions = RegionTable(
                    reg_code=np.asarray(f["regions/reg_code"][:]),
                    name=np.asarray(f["regions/name"][:]),
                )

            fields: dict[str, np.ndarray] = {}
            for key in ("a", "bx", "by", "j"):
                h5_key = f"fields/{key}"
                if h5_key in f:
                    fields[key] = np.asarray(f[h5_key][:])

            attrs = {"source_path": str(path)}
            if "steps" in f:
                attrs["steps"] = np.asarray(f["steps"][:])

        return MeshFrame(
            topology=topology,
            nodes=nodes,
            regions=regions,
            fields=fields,
            attrs=attrs,
        )
